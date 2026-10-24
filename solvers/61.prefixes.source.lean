import «.formalsw».solvers.«61.py.ast»
import lang.py.FormalSW.HeapCalls
import lang.py.FormalSW.KeyDictionaryCalls

set_option Elab.async false

namespace solvers.«61».build_prefix_map
open FormalSW.PyAst FormalSW.PyAst.HeapCalls

private abbrev program := Solvers.P61Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
private def numbers (values : List Int) : List Mutable.Value := values.map number

def bucket (values : List Int) (key : Int) : List Int :=
  values.filter (fun value => decide (value / 100 = key))

def lookup : List (Int × Nat) → Int → Option Nat
  | [], _ => none
  | (stored, address) :: rest, key => if stored = key then some address else lookup rest key

private def encode (entries : List (Int × Nat)) : List (Constant × Mutable.Value) :=
  entries.map (fun (key, address) => (.int key, .ref address))

private theorem lookup_encode (entries : List (Int × Nat)) (key : Int) :
    ScalarDict.lookup (encode entries) (.int key) = (lookup entries key).map Mutable.Value.ref := by
  induction entries with
  | nil => rfl
  | cons entry rest ih =>
      obtain ⟨stored, address⟩ := entry
      by_cases same : stored = key
      · simp [encode, lookup, ScalarDict.lookup, scalarCompare, evalCmpOp, same]
      · simpa [encode, lookup, ScalarDict.lookup, scalarCompare, evalCmpOp, same] using ih

private theorem lookup_append (left right : List (Int × Nat)) (key : Int) :
    lookup (left ++ right) key =
      match lookup left key with
      | some address => some address
      | none => lookup right key := by
  induction left with
  | nil => rfl
  | cons entry rest ih =>
      obtain ⟨stored, address⟩ := entry
      by_cases same : stored = key <;> simp [lookup, same, ih]

private theorem insert_encode (entries : List (Int × Nat)) (key : Int) (address : Nat)
    (absent : lookup entries key = none) :
    ScalarDict.insert (encode entries) (.int key) (.ref address) = encode (entries ++ [(key, address)]) := by
  induction entries with
  | nil => rfl
  | cons entry rest ih =>
      obtain ⟨stored, previous⟩ := entry
      by_cases same : stored = key
      · simp [lookup, same] at absent
      · have tail : lookup rest key = none := by simpa [lookup, same] using absent
        simpa [encode, ScalarDict.insert, scalarCompare, evalCmpOp, same] using
          congrArg (fun result : List (Constant × Mutable.Value) => (.int stored, .ref previous) :: result) (ih tail)

private theorem found_append (heap extra : Mutable.Heap) (address : Nat) (object : Mutable.Object)
    (found : heap[address]? = some object) : (heap ++ extra)[address]? = some object := by
  rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1]
  exact found

private theorem extension_set (base heap : Mutable.Heap)
    (extension : ∃ extra, heap = base ++ extra) (address : Nat) (outside : base.length ≤ address)
    (object : Mutable.Object) : ∃ extra, heap.set address object = base ++ extra := by
  obtain ⟨extra, rfl⟩ := extension
  exact ⟨extra.set (address - base.length) object, List.set_append_right address object outside⟩

/-- The source table owns distinct fresh list objects. Its buckets retain input order. -/
structure Store (base heap : Mutable.Heap) (processed : List Int) where
  entries : List (Int × Nat)
  extension : ∃ extra, heap = base ++ extra
  table : heap[base.length]? = some (.dict (encode entries))
  present : ∀ key address, lookup entries key = some address →
    base.length < address ∧ heap[address]? = some (.list (numbers (bucket processed key)))
  exclusive : ∀ left right address,
    lookup entries left = some address → lookup entries right = some address → left = right
  missing : ∀ key, lookup entries key = none → bucket processed key = []

def Store.initial (base : Mutable.Heap) : Store base (base ++ [.dict []]) [] where
  entries := []
  extension := ⟨[.dict []], rfl⟩
  table := by simp [encode]
  present := by intro key address found; simp [lookup] at found
  exclusive := by intro left right address found; simp [lookup] at found
  missing := by intro key _; rfl

variable {base heap : Mutable.Heap} {processed : List Int}

def Store.append (store : Store base heap processed) (extra : Mutable.Heap) :
    Store base (heap ++ extra) processed where
  entries := store.entries
  extension := by
    obtain ⟨old, rfl⟩ := store.extension
    exact ⟨old ++ extra, List.append_assoc _ _ _⟩
  table := found_append heap extra _ _ store.table
  present := by
    intro key address looked
    obtain ⟨outside, found⟩ := store.present key address looked
    exact ⟨outside, found_append heap extra address _ found⟩
  exclusive := store.exclusive
  missing := store.missing

/-- Eager evaluation allocates a default even if another bucket already exists. -/
def Store.add_empty (store : Store base heap processed) (key : Int) (absent : lookup store.entries key = none) :
    Store base ((heap ++ [Mutable.Object.list []]).set base.length
      (.dict (encode (store.entries ++ [(key, heap.length)])))) processed := by
  let entries : List (Int × Nat) := store.entries ++ [(key, heap.length)]
  let extended : Mutable.Heap := heap ++ [.list []]
  let updated : Mutable.Heap := extended.set base.length (.dict (encode entries))
  have tableInside : base.length < heap.length := (List.getElem?_eq_some_iff.mp store.table).1
  have classify (query : Int) (address : Nat) (looked : lookup entries query = some address) :
      lookup store.entries query = some address ∨ (query = key ∧ address = heap.length) := by
    cases old : lookup store.entries query with
    | some previous =>
        have same : previous = address := by simpa [entries, lookup_append, old] using looked
        exact Or.inl (by simp [same])
    | none =>
        by_cases same : key = query
        · subst query
          have fresh : heap.length = address := by simpa [entries, lookup_append, old, lookup] using looked
          exact Or.inr ⟨rfl, fresh.symm⟩
        · simp [entries, lookup_append, old, lookup, same] at looked
  refine { entries := entries, extension := ?_, table := ?_, present := ?_, exclusive := ?_, missing := ?_ }
  · exact extension_set base extended (store.append [.list []]).extension _ (Nat.le_refl _) _
  · exact List.getElem?_set_self (by simp only [List.length_append, List.length_singleton]; omega)
  · intro query address looked
    rcases classify query address looked with old | ⟨rfl, rfl⟩
    · obtain ⟨outside, found⟩ := store.present query address old
      refine ⟨outside, ?_⟩
      simpa only [updated, List.getElem?_set_ne (by omega : base.length ≠ address)] using
        found_append heap [.list []] address _ found
    · refine ⟨tableInside, ?_⟩
      have empty := store.missing _ absent
      rw [List.getElem?_set_ne (by omega : base.length ≠ heap.length)]
      simp [empty, numbers]
  · intro left right address leftFound rightFound
    rcases classify left address leftFound with oldLeft | ⟨rfl, freshLeft⟩
    · rcases classify right address rightFound with oldRight | ⟨rfl, freshRight⟩
      · exact store.exclusive left right address oldLeft oldRight
      · have found := (store.present left address oldLeft).2
        have inside := (List.getElem?_eq_some_iff.mp found).1
        omega
    · rcases classify right address rightFound with oldRight | ⟨rfl, _⟩
      · have found := (store.present right address oldRight).2
        have inside := (List.getElem?_eq_some_iff.mp found).1
        omega
      · rfl
  · intro query missing
    have old : lookup store.entries query = none := by
      cases looked : lookup store.entries query with
      | none => rfl
      | some address => simp [entries, lookup_append, looked] at missing
    exact store.missing query old

/-- Mutating one bucket preserves the table, all other buckets, and the caller prefix. -/
def Store.bump (store : Store base heap processed) (value : Int) (address : Nat)
    (looked : lookup store.entries (value / 100) = some address) :
    Store base (heap.set address (.list (numbers (bucket processed (value / 100) ++ [value]))))
      (processed ++ [value]) := by
  obtain ⟨outside, found⟩ := store.present (value / 100) address looked
  refine { entries := store.entries
           extension := ?_
           table := ?_
           present := ?_
           exclusive := store.exclusive
           missing := ?_ }
  · exact extension_set base heap store.extension address (by omega) _
  · simpa only [List.getElem?_set_ne (by omega : address ≠ base.length)] using store.table
  · intro key other otherLooked
    obtain ⟨otherOutside, otherFound⟩ := store.present key other otherLooked
    refine ⟨otherOutside, ?_⟩
    by_cases same : key = value / 100
    · subst key
      have equal : other = address := Option.some.inj (otherLooked.symm.trans looked)
      subst other
      simpa [bucket] using List.getElem?_set_self (a := .list (numbers (bucket processed (value / 100) ++ [value])))
        (List.getElem?_eq_some_iff.mp found).1
    · have separate : address ≠ other := by
        intro equal
        subst other
        exact same (store.exclusive key (value / 100) address otherLooked looked)
      have different : value / 100 ≠ key := Ne.symm same
      simpa [bucket, different, List.getElem?_set_ne separate] using otherFound
  · intro key absent
    have different : value / 100 ≠ key := by
      intro equal
      subst key
      rw [looked] at absent
      cases absent
    simpa [bucket, different] using store.missing key absent

/-- Search callers need the bucket contents after the construction frame has returned. -/
structure Buckets (heap : Mutable.Heap) (address : Nat) (values : List Int) where
  entries : List (Int × Nat)
  table : heap[address]? = some (.dict (encode entries))
  present : ∀ key target, lookup entries key = some target →
    address < target ∧ heap[target]? = some (.list (numbers (bucket values key)))
  missing : ∀ key, lookup entries key = none → bucket values key = []

def Store.view (store : Store base heap processed) : Buckets heap base.length processed where
  entries := store.entries
  table := store.table
  present := store.present
  missing := store.missing

variable {address : Nat} {values : List Int}

def Buckets.append (view : Buckets heap address values) (extra : Mutable.Heap) :
    Buckets (heap ++ extra) address values where
  entries := view.entries
  table := found_append heap extra address _ view.table
  present := by
    intro key target looked
    obtain ⟨later, found⟩ := view.present key target looked
    exact ⟨later, found_append heap extra target _ found⟩
  missing := view.missing

/-- Storing an outer dictionary entry cannot change a later nested prefix map. -/
def Buckets.set_before (view : Buckets heap address values) (location : Nat) (earlier : location < address)
    (object : Mutable.Object) : Buckets (heap.set location object) address values where
  entries := view.entries
  table := by simpa only [List.getElem?_set_ne (by omega : location ≠ address)] using view.table
  present := by
    intro key target looked
    obtain ⟨later, found⟩ := view.present key target looked
    exact ⟨later, by simpa only [List.getElem?_set_ne (by omega : location ≠ target)] using found⟩
  missing := view.missing

private def defaultExpression : Expr :=
  .call (.attribute (.name "mp") "setdefault") [.name "pref", .list []]

private theorem default_evaluates (store : Store base heap processed) (key : Int) (env : HeapCalls.Env)
    (getMap : env "mp" = some (.ref base.length)) (getKey : env "pref" = some (number key))
    (globals : Mutable.Globals) :
    ∃ (middle : Mutable.Heap) (address : Nat) (ready : Store base middle processed),
      lookup ready.entries key = some address ∧
      Evaluates program env defaultExpression heap (.ref address) middle globals := by
  have table := found_append heap [.list []] base.length _ store.table
  cases old : lookup store.entries key with
  | none =>
      let middle : Mutable.Heap := (heap ++ [Mutable.Object.list []]).set base.length
        (.dict (encode (store.entries ++ [(key, heap.length)])))
      let ready := store.add_empty key old
      refine ⟨middle, heap.length, ready, ?_, ?_⟩
      · change lookup (store.entries ++ [(key, heap.length)]) key = some heap.length
        simp [lookup_append, old, lookup]
      · apply Evaluates.dictionary_setdefault (address := base.length) (afterReceiver := heap)
          (afterArguments := heap ++ [.list []]) (values := [number key, .ref heap.length])
          (by intro name same; cases same; rw [getMap]; simp) (Evaluates.lookup getMap)
        · simp [bindMethod, store.table]
        · exact .cons (Evaluates.lookup getKey) (.cons (Evaluates.list .nil) .nil)
        · change Mutable.dictionarySetDefault base.length (number key) (.ref heap.length)
            (heap ++ [.list []]) = _
          rw [Mutable.dictionarySetDefault_scalar _ _ _ table (.int key) (.ref heap.length), lookup_encode, old]
          simp [middle, insert_encode _ _ _ old]
  | some address =>
      let ready := store.append [.list []]
      refine ⟨heap ++ [.list []], address, ready, old, ?_⟩
      apply Evaluates.dictionary_setdefault (address := base.length) (afterReceiver := heap)
        (afterArguments := heap ++ [.list []]) (values := [number key, .ref heap.length])
        (by intro name same; cases same; rw [getMap]; simp) (Evaluates.lookup getMap)
      · simp [bindMethod, store.table]
      · exact .cons (Evaluates.lookup getKey) (.cons (Evaluates.list .nil) .nil)
      · change Mutable.dictionarySetDefault base.length (number key) (.ref heap.length)
          (heap ++ [.list []]) = _
        rw [Mutable.dictionarySetDefault_scalar _ _ _ table (.int key) (.ref heap.length), lookup_encode, old]
        rfl

private def appendCode : List Stmt :=
  [.expr (.call (.attribute defaultExpression "append") [.name "x"])]

private theorem emit (store : Store base heap processed) (value : Int) (env : HeapCalls.Env)
    (getMap : env "mp" = some (.ref base.length)) (getKey : env "pref" = some (number (value / 100)))
    (getValue : env "x" = some (number value)) (globals : Mutable.Globals) :
    ∃ (finalHeap : Mutable.Heap) (_finalStore : Store base finalHeap (processed ++ [value])),
      BlockContinues program env appendCode heap env finalHeap globals := by
  obtain ⟨middle, address, ready, looked, prepared⟩ := default_evaluates store (value / 100) env getMap getKey globals
  have found := (ready.present (value / 100) address looked).2
  refine ⟨_, ready.bump value address looked, ?_⟩
  simpa only [numbers, List.map_append, List.map_cons, List.map_nil] using
    BlockContinues.discard (Evaluates.append prepared found (Evaluates.lookup getValue) found)

private def body : List Stmt :=
  [.assign [.name "pref"] (.binOp (.name "x") .floorDiv (.constant (.int 100)))] ++ appendCode
private def annotation : Expr :=
  .subscript (.name "dict") (.tuple [.name "int", .subscript (.name "list") (.name "int")])
private def code : List Stmt :=
  [.annAssign (.name "mp") annotation (some (.dict [])),
   .forIn (.name "x") (.name "nums") body [], .return (.name "mp")]

theorem code_source :
    executableBody ((findFunctionIn program.body "build_prefix_map").getD ([], [])).2 = code := rfl

private theorem source_found (store : Store base heap processed) (address : Nat) (values : List Int)
    (found : base[address]? = some (.list (numbers values))) :
    heap[address]? = some (.list (numbers values)) := by
  obtain ⟨extra, rfl⟩ := store.extension
  exact found_append base extra address _ found

/-- The actual source walks the original list and appends each number to its prefix bucket. -/
theorem loop (remaining values : List Int) (position : Nat) (suffix : values.drop position = remaining)
    (store : Store base heap processed) (input : Nat) (found : base[input]? = some (.list (numbers values)))
    (env : HeapCalls.Env) (getMap : env "mp" = some (.ref base.length)) (globals : Mutable.Globals := []) :
    ∃ (finalHeap : Mutable.Heap) (_finalStore : Store base finalHeap (processed ++ remaining)),
      BlockReturns program env
        [.sequenceStepLoop "x" (.ref input) position body [], .return (.name "mp")]
        heap (.ref base.length) finalHeap globals := by
  induction remaining generalizing position heap processed env with
  | nil =>
      refine ⟨heap, ?_, ?_⟩
      · simpa only [List.append_nil] using store
      · apply BlockReturns.sequence_done (items := numbers values)
          (by simp [sequenceValue, Mutable.sequence, source_found store input values found])
        · have done : values[position]? = none := by rw [← List.head?_drop, suffix]; rfl
          simp [numbers, List.getElem?_map, done]
        · exact BlockReturns.return_value (Evaluates.lookup getMap)
  | cons value rest ih =>
      let bound := setLocal env "x" (number value)
      let prefixed := setLocal bound "pref" (number (value / 100))
      have boundMap : bound "mp" = some (.ref base.length) := by simpa [bound, setLocal] using getMap
      have prefixedMap : prefixed "mp" = some (.ref base.length) := by simpa [prefixed, setLocal] using boundMap
      have prefixRun : BlockContinues program bound
          [.assign [.name "pref"] (.binOp (.name "x") .floorDiv (.constant (.int 100)))] heap prefixed heap globals := by
        apply BlockContinues.assign
        apply Evaluates.binary (middle := heap) (afterRight := heap) (finalHeap := heap)
          (av := number value) (bv := number 100) (Evaluates.lookup rfl) Evaluates.constant
        simp [Mutable.binary, scalarBinary, number]
      obtain ⟨middle, ready, emitted⟩ := emit store value prefixed prefixedMap rfl
        (by simp [prefixed, bound, setLocal]) globals
      obtain ⟨finalHeap, finishedStore, executed⟩ := ih (position + 1)
        (by simp only [List.drop_add_one_eq_tail_drop, suffix, List.tail_cons]) ready prefixed prefixedMap
      refine ⟨finalHeap, ?_, ?_⟩
      · simpa only [List.append_assoc, List.singleton_append] using finishedStore
      · apply BlockReturns.sequence_step (items := numbers values) (value := number value)
          (by simp [sequenceValue, Mutable.sequence, source_found store input values found])
        · have item : values[position]? = some value := by rw [← List.head?_drop, suffix]; rfl
          simp [numbers, List.getElem?_map, item]
        · simpa [body, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, appendCode,
            List.append_assoc] using (prefixRun.append emitted).returns executed

/-- Arbitrary integer inputs produce exactly their prefix buckets in fresh heap objects. -/
theorem result (values : List Int) (heap : Mutable.Heap) (input : Nat)
    (found : heap[input]? = some (.list (numbers values))) (globals : Mutable.Globals := []) :
    ∃ (finalHeap : Mutable.Heap) (_finalStore : Store heap finalHeap values),
      ReturnsValue program "build_prefix_map" [.ref input] heap (.ref heap.length) finalHeap globals := by
  let params := ((findFunctionIn program.body "build_prefix_map").getD ([], [])).1
  let sourceBody := ((findFunctionIn program.body "build_prefix_map").getD ([], [])).2
  let env := functionEnv params sourceBody [.ref input] globals
  let initialized := setLocal env "mp" (.ref heap.length)
  let initial := Store.initial heap
  obtain ⟨finalHeap, finalStore, executed⟩ := loop values values 0 rfl initial input found initialized rfl globals
  refine ⟨finalHeap, by simpa using finalStore, ?_⟩
  apply BlockReturns.function rfl rfl
  change BlockReturns program env (executableBody sourceBody) heap (.ref heap.length) finalHeap globals
  rw [code_source]
  have allocated : BlockContinues program env [.annAssign (.name "mp") annotation (some (.dict []))]
      heap initialized (heap ++ [.dict []]) globals := BlockContinues.annAssign (Evaluates.dictionary .nil)
  apply allocated.returns
  apply BlockReturns.for_sequence (items := numbers values) (by intros; intro equal; cases equal)
    (Evaluates.lookup (by cases globals <;> rfl))
    (by simp [Mutable.sequence, source_found initial input values found])
  exact executed

end solvers.«61».build_prefix_map
