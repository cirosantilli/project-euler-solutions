import «.formalsw».solvers.«54.py.ast»
import lang.py.FormalSW.DictionaryCalls
import lang.py.FormalSW.DictionaryItemsCalls
import lang.py.FormalSW.ReverseSortingCalls
import lang.py.FormalSW.TupleComparisons
import ProjectEulerSolutions.Lib.HeapCollection
import ProjectEulerSolutions.Lib.PokerGroupOrder
import ProjectEulerSolutions.Lib.TupleSorting
import Mathlib.Data.List.Forall2

/-! Source proofs for poker-card parsing and hand evaluation. -/
set_option Elab.async false
namespace solvers.«54»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
private abbrev program := Solvers.P54Ast.module
private abbrev number (n : Int) : Mutable.Value := .scalar (.int n)
private abbrev flag (b : Bool) : Mutable.Value := .scalar (.bool b)
private abbrev character (c : Char) : Mutable.Value := .scalar (.string (String.singleton c))

namespace count_values

/-- Ordered integer counts retain the first position at which each key appeared. -/
def lookup : List (Int × Int) → Int → Int
  | [], _ => 0
  | (stored, value) :: rest, key => if stored = key then value else lookup rest key

def put : List (Int × Int) → Int → Int → List (Int × Int)
  | [], key, value => [(key, value)]
  | (stored, previous) :: rest, key, value =>
      if stored = key then (stored, value) :: rest else (stored, previous) :: put rest key value

def bump (table : List (Int × Int)) (key : Int) : List (Int × Int) := put table key (lookup table key + 1)

def counts (values : List Int) : List (Int × Int) := values.foldl bump []

def entries (table : List (Int × Int)) : List (Constant × Mutable.Value) :=
  table.map fun (key, value) => (.int key, number value)

private abbrev numbers (values : List Int) : List Mutable.Value := values.map number
private abbrev working (heap : Mutable.Heap) (table : List (Int × Int)) : Mutable.Heap := heap ++ [.dict (entries table)]

theorem lookup_put (table : List (Int × Int)) (key value query : Int) :
    lookup (put table key value) query = if query = key then value else lookup table query := by
  induction table with
  | nil => by_cases same : query = key <;> simp [put, lookup, same, Ne.symm]
  | cons pair rest ih =>
      obtain ⟨stored, previous⟩ := pair
      by_cases updated : stored = key <;> by_cases selected : stored = query <;>
        by_cases same : query = key <;> simp_all [put, lookup]

theorem lookup_bump (table : List (Int × Int)) (key query : Int) :
    lookup (bump table key) query = lookup table query + if query = key then 1 else 0 := by
  rw [bump, lookup_put]
  by_cases same : query = key <;> simp [same]

private theorem lookup_fold (values : List Int) (table : List (Int × Int)) (query : Int) :
    lookup (values.foldl bump table) query = lookup table query + (values.count query : Int) := by
  induction values generalizing table with
  | nil => simp
  | cons value rest ih =>
      rw [List.foldl_cons, ih, lookup_bump]
      by_cases same : query = value
      · simp [same]
        omega
      · simp [same, Ne.symm]

/-- The dictionary lookup, defaulting to zero, equals the exact number of occurrences. -/
theorem lookup_counts (values : List Int) (query : Int) :
    lookup (counts values) query = (values.count query : Int) := by
  simpa [counts, lookup] using lookup_fold values [] query

private theorem entries_put (table : List (Int × Int)) (key value : Int) :
    entries (put table key value) = ScalarDict.insert (entries table) (.int key) (number value) := by
  induction table with
  | nil => rfl
  | cons pair rest ih =>
      obtain ⟨stored, previous⟩ := pair
      by_cases same : stored = key
      · simp [put, entries, ScalarDict.insert, scalarCompare, evalCmpOp, same]
      · simpa [put, entries, ScalarDict.insert, scalarCompare, evalCmpOp, same] using
          congrArg (List.cons (Constant.int stored, number previous)) ih

private theorem lookup_entries (table : List (Int × Int)) (key : Int) :
    (ScalarDict.lookup (entries table) (.int key)).getD (number 0) = number (lookup table key) := by
  induction table with
  | nil => rfl
  | cons pair rest ih =>
      obtain ⟨stored, value⟩ := pair
      by_cases same : stored = key
      · simp [entries, ScalarDict.lookup, lookup, scalarCompare, evalCmpOp, same]
      · simpa [entries, ScalarDict.lookup, lookup, scalarCompare, evalCmpOp, same] using ih

private def body : List Stmt :=
  [.assign [.subscript (.name "counts") (.name "value")]
    (.binOp (.call (.attribute (.name "counts") "get") [.name "value", .constant (.int 0)]) .add (.constant (.int 1)))]

private theorem input_found (heap : Mutable.Heap) (address : Nat) (values : List Int)
    (found : heap[address]? = some (.list (numbers values))) (table : List (Int × Int)) :
    (working heap table)[address]? = some (.list (numbers values)) := by
  rw [working, List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1]
  exact found

private theorem increment (heap : Mutable.Heap) (table : List (Int × Int)) (value : Int)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getCounts : env "counts" = some (.ref heap.length)) (getValue : env "value" = some (number value)) :
    BlockContinues program env body (working heap table) env (working heap (bump table value)) globals := by
  apply BlockContinues.setItem (by intro a b c impossible; cases impossible)
    (afterRhs := working heap table) (afterTarget := working heap table)
    (value := number (lookup table value + 1))
  · apply Evaluates.binary (middle := working heap table) (afterRight := working heap table)
      (av := number (lookup table value)) (bv := number 1)
    · apply Evaluates.dictionary_get (entries := entries table)
        (by intro name equal; cases equal; simp [getCounts])
        (Evaluates.lookup getCounts) (by simp [working])
        (.cons (Evaluates.lookup getValue) (.cons Evaluates.constant .nil))
      simpa only [lookup_entries] using
        Mutable.dictionaryGet_scalar (working heap table) heap.length (entries table) (by simp [working]) (.int value) (number 0)
    · exact Evaluates.constant
    · simp [Mutable.binary, scalarBinary, number]
  · exact .cons (Evaluates.lookup getCounts) (.cons (Evaluates.lookup getValue) .nil)
  · simpa [working, bump, entries_put, List.set_append_right] using
      Mutable.setItem_dictionary (working heap table) heap.length (entries table) (by simp [working])
        (.int value) (number (lookup table value + 1))

private theorem scan (heap : Mutable.Heap) (address : Nat) (values : List Int)
    (found : heap[address]? = some (.list (numbers values))) (remaining before : List Int)
    (decomposition : values = before ++ remaining) (table : List (Int × Int))
    (env : HeapCalls.Env) (globals : Mutable.Globals) (getCounts : env "counts" = some (.ref heap.length)) :
    BlockReturns program env
      [.sequenceStepLoop "value" (.ref address) before.length body [], .return (.name "counts")]
      (working heap table) (.ref heap.length) (working heap (remaining.foldl bump table)) globals := by
  induction remaining generalizing before table env with
  | nil =>
      apply BlockReturns.sequence_done (items := numbers values)
      · simp [sequenceValue, Mutable.sequence, input_found heap address values found table]
      · simp [numbers, decomposition]
      · exact BlockReturns.return_value (Evaluates.lookup getCounts)
  | cons value rest ih =>
      let nextEnv := setLocal env "value" (number value)
      apply BlockReturns.sequence_step (items := numbers values) (value := number value)
      · simp [sequenceValue, Mutable.sequence, input_found heap address values found table]
      · simp [numbers, decomposition]
      · apply (increment heap table value nextEnv globals (by simpa [nextEnv, setLocal] using getCounts) rfl).returns
        change BlockReturns program nextEnv
          [.sequenceStepLoop "value" (.ref address) (before.length+1) body [], .return (.name "counts")]
          (working heap (bump table value)) (.ref heap.length) (working heap (rest.foldl bump (bump table value))) globals
        have nextDecomposition : values = (before ++ [value]) ++ rest := by
          simpa only [List.append_assoc, List.singleton_append] using decomposition
        simpa only [List.length_append, List.length_singleton] using
          ih (before ++ [value]) nextDecomposition (bump table value) nextEnv
            (by simpa [nextEnv, setLocal] using getCounts)

/-- Counting leaves the complete caller heap intact and returns one fresh dictionary. -/
theorem result (heap : Mutable.Heap) (address : Nat) (values : List Int) (globals : Mutable.Globals)
    (found : heap[address]? = some (.list (numbers values))) :
    ReturnsValue program "count_values" [.ref address] heap (.ref heap.length)
      (working heap (counts values)) globals := by
  let entry := functionEnv ((findFunctionIn program.body "count_values").getD ([], [])).1
    ((findFunctionIn program.body "count_values").getD ([], [])).2 [.ref address] globals
  let ready := setLocal entry "counts" (.ref heap.length)
  have getValues : ready "values" = some (.ref address) := by cases globals <;> rfl
  apply BlockReturns.function rfl rfl
  apply (BlockContinues.annAssign (Evaluates.dictionary .nil)).returns
  apply BlockReturns.for_sequence (items := numbers values)
    (by intro expressions impossible; cases impossible) (Evaluates.lookup getValues)
  · have preserved : (heap ++ [Mutable.Object.dict []])[address]? = some (.list (numbers values)) :=
      input_found heap address values found []
    simp [Mutable.sequence, preserved]
  · exact scan heap address values found values [] (by simp) [] ready globals rfl

theorem «Returns» (heap : Mutable.Heap) (address : Nat) (_valid : heap.Valid)
    (typed : Mutable.Value.HasType heap (.ref address) "list[int]") :
    (HeapCalls.call program "count_values" [.ref address] heap).Returns := by
  obtain ⟨values, found⟩ := Mutable.Value.ints_of_hasType typed
  exact (result heap address values [] found).returns

theorem «Supported» (heap : Mutable.Heap) (address : Nat) (valid : heap.Valid)
    (typed : Mutable.Value.HasType heap (.ref address) "list[int]") :
    (HeapCalls.call program "count_values" [.ref address] heap).Supported := («Returns» heap address valid typed).supported


/-- Updating the first matching key changes the stored mass by its replacement difference. -/
theorem sum_put (table : List (Int × Int)) (key value : Int) :
    ((put table key value).map Prod.snd).sum = (table.map Prod.snd).sum + value - lookup table key := by
  induction table with
  | nil => simp [put, lookup]
  | cons pair rest ih =>
      obtain ⟨stored, previous⟩ := pair
      by_cases same : stored = key
      · rw [put, if_pos same, lookup, if_pos same]
        simp only [List.map_cons, List.sum_cons]
        omega
      · rw [put, if_neg same, lookup, if_neg same]
        simp only [List.map_cons, List.sum_cons]
        rw [ih]
        omega

theorem sum_bump (table : List (Int × Int)) (key : Int) :
    ((bump table key).map Prod.snd).sum = (table.map Prod.snd).sum + 1 := by
  rw [bump, sum_put]
  omega

private theorem sum_fold (values : List Int) (table : List (Int × Int)) :
    ((values.foldl bump table).map Prod.snd).sum = (table.map Prod.snd).sum + (values.length : Int) := by
  induction values generalizing table with
  | nil => simp
  | cons value rest ih =>
      rw [List.foldl_cons, ih, sum_bump]
      simp only [List.length_cons]
      omega

/-- Stored frequencies account for every processed value exactly once. -/
theorem counts_sum (values : List Int) :
    ((counts values).map Prod.snd).sum = (values.length : Int) := by
  simpa [counts] using sum_fold values []

theorem lookup_nonnegative (table : List (Int × Int)) (key : Int)
    (positive : ∀ entry ∈ table, 0 < entry.2) : 0 ≤ lookup table key := by
  revert positive
  induction table with
  | nil => intro _; simp [lookup]
  | cons pair rest ih =>
      obtain ⟨stored, previous⟩ := pair
      intro positive
      have first := positive (stored, previous) List.mem_cons_self
      have tail := ih (fun entry member => positive entry (List.mem_cons_of_mem _ member))
      by_cases same : stored = key
      · rw [lookup, if_pos same]
        omega
      · rw [lookup, if_neg same]
        exact tail

theorem put_positive (table : List (Int × Int)) (key value : Int) (newPositive : 0 < value)
    (positive : ∀ entry ∈ table, 0 < entry.2) :
    ∀ entry ∈ put table key value, 0 < entry.2 := by
  revert positive
  induction table with
  | nil => intro _; simpa [put] using newPositive
  | cons pair rest ih =>
      obtain ⟨stored, previous⟩ := pair
      intro positive
      have first := positive (stored, previous) List.mem_cons_self
      have tail : ∀ entry ∈ rest, 0 < entry.2 :=
        fun entry member => positive entry (List.mem_cons_of_mem _ member)
      by_cases same : stored = key
      · rw [put, if_pos same]
        exact List.forall_mem_cons.mpr ⟨newPositive, tail⟩
      · rw [put, if_neg same]
        exact List.forall_mem_cons.mpr ⟨first, ih tail⟩

theorem bump_positive (table : List (Int × Int)) (key : Int)
    (positive : ∀ entry ∈ table, 0 < entry.2) :
    ∀ entry ∈ bump table key, 0 < entry.2 := by
  apply put_positive table key (lookup table key + 1) (by
    have nonnegative := lookup_nonnegative table key positive
    omega) positive

private theorem positive_fold (values : List Int) (table : List (Int × Int))
    (positive : ∀ entry ∈ table, 0 < entry.2) :
    ∀ entry ∈ values.foldl bump table, 0 < entry.2 := by
  induction values generalizing table with
  | nil => exact positive
  | cons value rest ih =>
      rw [List.foldl_cons]
      exact ih (bump table value) (bump_positive table value positive)

/-- Every stored frequency is strictly positive; unseen keys are absent. -/
theorem counts_positive (values : List Int) : ∀ entry ∈ counts values, 0 < entry.2 :=
  positive_fold values [] (by intro entry impossible; cases impossible)

theorem mem_keys_put (table : List (Int × Int)) (key value query : Int) :
    query ∈ (put table key value).map Prod.fst ↔ query = key ∨ query ∈ table.map Prod.fst := by
  induction table with
  | nil => simp [put]
  | cons pair rest ih =>
      obtain ⟨stored, previous⟩ := pair
      by_cases same : stored = key
      · subst stored
        simp [put]
      · rw [put, if_neg same]
        simp only [List.map_cons, List.mem_cons, ih, or_left_comm]

theorem keys_nodup_put (table : List (Int × Int)) (key value : Int)
    (unique : (table.map Prod.fst).Nodup) : ((put table key value).map Prod.fst).Nodup := by
  revert unique
  induction table with
  | nil => intro _; simp [put]
  | cons pair rest ih =>
      obtain ⟨stored, previous⟩ := pair
      intro unique
      have both : stored ∉ rest.map Prod.fst ∧ (rest.map Prod.fst).Nodup := by
        simpa only [List.map_cons, List.nodup_cons] using unique
      by_cases same : stored = key
      · rw [put, if_pos same]
        simpa only [List.map_cons] using unique
      · rw [put, if_neg same]
        apply List.nodup_cons.mpr
        refine ⟨?_, ih both.2⟩
        intro member
        exact ((mem_keys_put rest key value stored).mp member).elim same both.1

private theorem unique_fold (values : List Int) (table : List (Int × Int))
    (unique : (table.map Prod.fst).Nodup) : ((values.foldl bump table).map Prod.fst).Nodup := by
  induction values generalizing table with
  | nil => exact unique
  | cons value rest ih =>
      rw [List.foldl_cons]
      exact ih (bump table value) (keys_nodup_put table value _ unique)

theorem counts_keys_nodup (values : List Int) : ((counts values).map Prod.fst).Nodup :=
  unique_fold values [] (by simp)

private theorem mem_keys_fold (values : List Int) (table : List (Int × Int)) (query : Int) :
    query ∈ (values.foldl bump table).map Prod.fst ↔ query ∈ values ∨ query ∈ table.map Prod.fst := by
  induction values generalizing table with
  | nil => simp
  | cons value rest ih =>
      rw [List.foldl_cons, ih, bump, mem_keys_put]
      simp only [List.mem_cons, or_assoc, or_left_comm]

/-- The dictionary stores exactly the ranks encountered in the input. -/
theorem mem_counts_keys (values : List Int) (query : Int) :
    query ∈ (counts values).map Prod.fst ↔ query ∈ values := by
  simpa [counts] using mem_keys_fold values [] query

end count_values

namespace rank_groups

private abbrev working := ProjectEulerSolutions.HeapCollection.working
private abbrev fields (table : List (Int × Int)) : List (Mutable.Value × Mutable.Value) :=
  (count_values.entries table).map fun entry => (.scalar entry.1, entry.2)
private abbrev keys (table : List (Int × Int)) : List SequenceSource :=
  (fields table).map fun entry => sequenceSource entry.1

def itemTuple (pair : Int × Int) : Mutable.Object := .tuple [number pair.1, number pair.2]
def groupTuple (pair : Int × Int) : Mutable.Object := .tuple [number pair.2, number pair.1]

/-- Each items iteration retains its original pair and its reversed ranking pair. -/
def scratch : List (Int × Int) → Mutable.Heap
  | [] => []
  | pair :: rest => itemTuple pair :: groupTuple pair :: scratch rest

def references (next : Nat) : List (Int × Int) → List Mutable.Value
  | [] => []
  | _ :: rest => .ref (next+1) :: references (next+2) rest

/-- Collector references have the same order as the dictionary's original entries. -/
def Represents (heap : Mutable.Heap) (groups : List Mutable.Value) (table : List (Int × Int)) : Prop :=
  List.Forall₂ (fun group pair => ∃ address : Nat, group = .ref address ∧ heap[address]? = some (groupTuple pair)) groups table

private theorem Represents.preserve (heap other : Mutable.Heap) (groups : List Mutable.Value)
    (table : List (Int × Int)) (rows : Represents heap groups table)
    (preserved : ∀ (address : Nat) (pair : Int × Int),
      heap[address]? = some (groupTuple pair) → other[address]? = some (groupTuple pair)) :
    Represents other groups table := by
  induction rows with
  | nil => exact .nil
  | @cons group pair groups table first rest ih =>
      obtain ⟨address, equal, found⟩ := first
      exact .cons ⟨address, equal, preserved address pair found⟩ ih

theorem Represents.typed (heap : Mutable.Heap) (groups : List Mutable.Value) (table : List (Int × Int))
    (rows : Represents heap groups table) :
    ∀ group ∈ groups, ∃ address : Nat, ∃ values : List Int,
      group = .ref address ∧ heap[address]? = some (.tuple (values.map number)) := by
  induction rows with
  | nil => intro group member; cases member
  | @cons group pair groups table first rest ih =>
      intro value member
      rcases List.mem_cons.mp member with equal | member
      · subst value
        obtain ⟨address, equal, found⟩ := first
        exact ⟨address, [pair.2, pair.1], equal, found⟩
      · exact ih value member

/-- Decoding the actual tuple references recovers each frequency/rank row in order. -/
theorem Represents.readInts (heap : Mutable.Heap) (groups : List Mutable.Value) (table : List (Int × Int))
    (rows : Represents heap groups table) :
    groups.map (ProjectEulerSolutions.TupleSorting.readInts heap) = table.map (fun pair => [pair.2, pair.1]) := by
  induction rows with
  | nil => rfl
  | @cons group pair groups table first rest ih =>
      obtain ⟨address, rfl, found⟩ := first
      simp only [List.map_cons, ih]
      congr 1
      simp [ProjectEulerSolutions.TupleSorting.readInts, found, groupTuple]

private theorem input_found (heap : Mutable.Heap) (address : Nat) (table : List (Int × Int))
    (found : heap[address]? = some (.dict (count_values.entries table))) (groups : List Mutable.Value) (garbage : Mutable.Heap) :
    (working heap groups garbage)[address]? = some (.dict (count_values.entries table)) :=
  ProjectEulerSolutions.HeapCollection.prefix_found heap (.list groups :: garbage) address _ found

private theorem rows_step (heap : Mutable.Heap) (groups : List Mutable.Value) (garbage : Mutable.Heap)
    (before : List (Int × Int)) (pair : Int × Int) (rows : Represents (working heap groups garbage) groups before) :
    Represents (working heap (groups ++ [.ref ((working heap groups garbage).length+1)])
      (garbage ++ [itemTuple pair, groupTuple pair]))
      (groups ++ [.ref ((working heap groups garbage).length+1)]) (before ++ [pair]) := by
  let extra := [itemTuple pair, groupTuple pair]
  let replacement := groups ++ [Mutable.Value.ref ((working heap groups garbage).length+1)]
  have kept : Represents (working heap replacement (garbage ++ extra)) groups before := by
    apply Represents.preserve (rows := rows)
    intro address row found
    have different : heap.length ≠ address := by
      intro same
      rw [← same, ProjectEulerSolutions.HeapCollection.found] at found
      cases found
    exact ProjectEulerSolutions.HeapCollection.other_found heap groups replacement garbage extra address _ different found
  have raw : (working heap groups (garbage ++ extra))[(working heap groups garbage).length+1]? = some (groupTuple pair) := by
    unfold working
    rw [← ProjectEulerSolutions.HeapCollection.append heap groups garbage extra]
    simp [extra]
  have different : heap.length ≠ (working heap groups garbage).length+1 := by
    simp only [ProjectEulerSolutions.HeapCollection.working, List.length_append, List.length_cons]
    omega
  have retained := ProjectEulerSolutions.HeapCollection.other_found heap groups replacement (garbage ++ extra) []
    ((working heap groups garbage).length+1) (groupTuple pair) different raw
  apply List.rel_append kept
  exact .cons ⟨_, rfl, by simpa only [List.append_nil] using retained⟩ .nil

private def target : Expr := .tuple [.name "value", .name "count"]
private def sourceBody : List Stmt :=
  [.expr (.call (.attribute (.name "groups") "append") [.tuple [.name "count", .name "value"]])]
private def loopBody : List Stmt := comprehensionTargetBody target sourceBody

private theorem emit (heap : Mutable.Heap) (groups : List Mutable.Value) (garbage : Mutable.Heap)
    (pair : Int × Int) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getGroups : env "groups" = some (.ref heap.length))
    (getPair : env "$formalsw.unpack" = some (.ref (working heap groups garbage).length)) :
    BlockContinues program env loopBody (working heap groups garbage ++ [itemTuple pair])
      (assignNames env ["value", "count"] [number pair.1, number pair.2])
      (working heap (groups ++ [.ref ((working heap groups garbage).length+1)])
        (garbage ++ [itemTuple pair, groupTuple pair])) globals := by
  let current := working heap groups garbage
  let ready := assignNames env ["value", "count"] [number pair.1, number pair.2]
  have unpacked := BlockContinues.unpack (module := program) (env := env) (globals := globals)
    (heap := current ++ [itemTuple pair]) (names := ["value", "count"]) (items := [number pair.1, number pair.2])
    (expressions := [.name "value", .name "count"])
    (value := .ref current.length) (finalHeap := current ++ [itemTuple pair])
    rfl (Evaluates.lookup getPair) (by simp [Mutable.sequence, itemTuple]) rfl
  apply BlockContinues.append unpacked
  apply BlockContinues.discard
  have getCount : ready "count" = some (number pair.2) := by simp [ready, assignNames, setLocal]
  have getValue : ready "value" = some (number pair.1) := by simp [ready, assignNames, setLocal]
  have getReady : ready "groups" = some (.ref heap.length) := by simpa [ready, assignNames, setLocal] using getGroups
  have tuple := Evaluates.tuple (module := program) (env := ready) (globals := globals)
    (heap := current ++ [itemTuple pair])
    (.cons (Evaluates.lookup getCount) (.cons (Evaluates.lookup getValue) .nil)) (by simp)
  have tupleEval : Evaluates program ready (.tuple [.name "count", .name "value"])
      (current ++ [itemTuple pair]) (.ref (current.length+1))
      (current ++ [itemTuple pair, groupTuple pair]) globals := by
    simpa only [groupTuple, List.length_append, List.length_singleton, List.append_assoc] using tuple
  have collector := ProjectEulerSolutions.HeapCollection.found heap groups garbage
  have beforeFound := ProjectEulerSolutions.HeapCollection.prefix_found current [itemTuple pair] heap.length _ collector
  have afterFound := ProjectEulerSolutions.HeapCollection.prefix_found current [itemTuple pair, groupTuple pair] heap.length _ collector
  have appended := Evaluates.append (module := program) (env := ready) (globals := globals)
    (Evaluates.lookup getReady) beforeFound tupleEval afterFound
  simpa only [current, ProjectEulerSolutions.HeapCollection.append, ProjectEulerSolutions.HeapCollection.store] using appended

private theorem scan (heap : Mutable.Heap) (address : Nat) (table : List (Int × Int))
    (found : heap[address]? = some (.dict (count_values.entries table)))
    (remaining before : List (Int × Int)) (decomposition : table = before ++ remaining)
    (groups : List Mutable.Value) (garbage : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getGroups : env "groups" = some (.ref heap.length)) (rows : Represents (working heap groups garbage) groups before) :
    ∃ finalEnv, finalEnv "groups" = some (.ref heap.length) ∧
      BlockContinues program env [.dictionaryItemsStepLoop "$formalsw.unpack" address (keys table) before.length loopBody []]
        (working heap groups garbage) finalEnv
        (working heap (groups ++ references (working heap groups garbage).length remaining) (garbage ++ scratch remaining)) globals ∧
      Represents (working heap (groups ++ references (working heap groups garbage).length remaining) (garbage ++ scratch remaining))
        (groups ++ references (working heap groups garbage).length remaining) (before ++ remaining) := by
  induction remaining generalizing before groups garbage env with
  | nil =>
      refine ⟨env, getGroups, ?_, ?_⟩
      · apply BlockContinues.dictionary_items_done (middle := working heap groups garbage)
        · have fetched := dictionaryItemsNext_fields (position := before.length)
            (input_found heap address table found groups garbage) (rfl : dictionaryItemFields (.dict (count_values.entries table)) = some (fields table))
          simpa [keys, fields, count_values.entries, decomposition] using fetched
        · simpa only [references, scratch, List.append_nil] using
            (BlockContinues.nil (module := program) (env := env) (heap := working heap groups garbage) (globals := globals))
      · simpa only [references, scratch, List.append_nil] using rows
  | cons pair rest ih =>
      let current := working heap groups garbage
      let pairedEnv := setLocal env "$formalsw.unpack" (.ref current.length)
      let ready := assignNames pairedEnv ["value", "count"] [number pair.1, number pair.2]
      let newGroups := groups ++ [Mutable.Value.ref (current.length+1)]
      let newGarbage := garbage ++ [itemTuple pair, groupTuple pair]
      have nextDecomposition : table = (before ++ [pair]) ++ rest := by
        simpa only [List.append_assoc, List.singleton_append] using decomposition
      have nextGet : ready "groups" = some (.ref heap.length) := by
        simpa [ready, pairedEnv, assignNames, setLocal] using getGroups
      have nextRows := rows_step heap groups garbage before pair rows
      obtain ⟨finalEnv, finalGet, continued, finalRows⟩ :=
        ih (before ++ [pair]) nextDecomposition newGroups newGarbage ready nextGet nextRows
      have size : (working heap newGroups newGarbage).length = current.length+2 := by
        simp only [current, newGarbage, ProjectEulerSolutions.HeapCollection.working, List.length_append, List.length_cons, List.length_nil]
        omega
      have referenceEq : newGroups ++ references (working heap newGroups newGarbage).length rest =
          groups ++ references current.length (pair :: rest) := by
        rw [size]
        simp only [newGroups, references, List.append_assoc, List.singleton_append]
      have garbageEq : newGarbage ++ scratch rest = garbage ++ scratch (pair :: rest) := by
        simp only [newGarbage, scratch, List.append_assoc, List.cons_append, List.nil_append]
      rw [referenceEq, garbageEq] at continued finalRows
      refine ⟨finalEnv, finalGet, ?_, ?_⟩
      · apply BlockContinues.dictionary_items_step (middle := current) (pair := .ref current.length)
          (key := number pair.1) (value := number pair.2) (afterTuple := current ++ [itemTuple pair])
        · have fetched := dictionaryItemsNext_fields (position := before.length)
            (input_found heap address table found groups garbage) (rfl : dictionaryItemFields (.dict (count_values.entries table)) = some (fields table))
          simpa [keys, fields, count_values.entries, decomposition] using fetched
        · simp [Mutable.allocateTuple, itemTuple, Except.map]
        · apply BlockContinues.append (emit heap groups garbage pair pairedEnv globals
            (by simpa [pairedEnv, setLocal] using getGroups) rfl)
          change BlockContinues program ready
            [.dictionaryItemsStepLoop "$formalsw.unpack" address (keys table) (before.length+1) loopBody []]
            (working heap newGroups newGarbage) finalEnv
            (working heap (groups ++ references current.length (pair :: rest)) (garbage ++ scratch (pair :: rest))) globals
          simpa only [List.length_append, List.length_singleton] using continued
      · simpa only [List.append_assoc, List.singleton_append] using finalRows

/-- Collector, item pairs, and ranking pairs occupy a fresh suffix of the caller's heap. -/
def collected (heap : Mutable.Heap) (table : List (Int × Int)) : Mutable.Heap :=
  working heap (references (heap.length+1) table) (scratch table)

def resultGroups (heap : Mutable.Heap) (table : List (Int × Int)) : List Mutable.Value :=
  ProjectEulerSolutions.TupleSorting.sorted (collected heap table) true (references (heap.length+1) table)

/-- The checked source loop establishes the row invariant for the complete collector. -/
theorem collected_represents (heap : Mutable.Heap) (address : Nat) (table : List (Int × Int))
    (found : heap[address]? = some (.dict (count_values.entries table))) :
    Represents (collected heap table) (references (heap.length+1) table) table := by
  let env : HeapCalls.Env := fun name => if name = "groups" then some (.ref heap.length) else none
  have executed := scan heap address table found table [] (by simp) [] [] env [] rfl .nil
  simp only [List.nil_append, ProjectEulerSolutions.HeapCollection.working,
    List.length_append, List.length_cons, List.length_nil, Nat.zero_add] at executed
  obtain ⟨_, _, _, rows⟩ := executed
  exact rows

/-- Sorting source references and decoding them agrees with sorting the numerical rows. -/
theorem resultGroups_readInts (heap : Mutable.Heap) (address : Nat) (table : List (Int × Int))
    (found : heap[address]? = some (.dict (count_values.entries table))) :
    (resultGroups heap table).map (ProjectEulerSolutions.TupleSorting.readInts (collected heap table)) =
      ProjectEulerSolutions.TupleSorting.sortedRows true (table.map (fun pair => [pair.2, pair.1])) := by
  rw [resultGroups, ProjectEulerSolutions.TupleSorting.sorted_readInts,
    Represents.readInts _ _ _ (collected_represents heap address table found)]

/-- Every tuple in the sorted result retains its contents after any further allocation. -/
theorem resultGroups_readInts_append (heap : Mutable.Heap) (address : Nat) (table : List (Int × Int))
    (found : heap[address]? = some (.dict (count_values.entries table))) (extra : Mutable.Heap) :
    (resultGroups heap table).map (ProjectEulerSolutions.TupleSorting.readInts (collected heap table ++ extra)) =
      (resultGroups heap table).map (ProjectEulerSolutions.TupleSorting.readInts (collected heap table)) := by
  apply List.map_congr_left
  intro group member
  have original := (ProjectEulerSolutions.TupleSorting.sorted_perm (collected heap table) true
    (references (heap.length+1) table)).mem_iff.mp member
  obtain ⟨location, values, rfl, tupleFound⟩ := Represents.typed _ _ _
    (collected_represents heap address table found) group original
  exact ProjectEulerSolutions.TupleSorting.readInts_append (collected heap table) extra location _ tupleFound

/-- The source's sorted group frequencies exhaust the possible five-card count profiles. -/
theorem resultGroups_profile (heap : Mutable.Heap) (address : Nat) (values : List Int)
    (found : heap[address]? = some (.dict (count_values.entries (count_values.counts values))))
    (five : values.length = 5) :
    ((resultGroups heap (count_values.counts values)).map
      (ProjectEulerSolutions.TupleSorting.readInts (collected heap (count_values.counts values)))).map
        (fun row => row.headD 0) ∈ ProjectEulerSolutions.PokerCountPatterns.profiles := by
  let pairs := (count_values.counts values).map fun pair => (pair.2, pair.1)
  have positive : ∀ pair ∈ pairs, 0 < pair.1 := by
    intro pair member
    obtain ⟨entry, entryMember, rfl⟩ := List.mem_map.mp member
    exact count_values.counts_positive values entry entryMember
  have total : (pairs.map Prod.fst).sum = 5 := by
    have mass := count_values.counts_sum values
    rw [five] at mass
    simpa only [pairs, List.map_map, Function.comp_def] using mass
  rw [resultGroups_readInts heap address (count_values.counts values) found]
  simpa only [pairs, ProjectEulerSolutions.PokerGroupOrder.row, List.map_map, Function.comp_def] using
    ProjectEulerSolutions.PokerGroupOrder.rows_profile pairs positive total

/-- The frequency profile remains valid in the final heap returned by rank_groups. -/
theorem resultGroups_profile_returned (heap : Mutable.Heap) (address : Nat) (values : List Int)
    (found : heap[address]? = some (.dict (count_values.entries (count_values.counts values))))
    (five : values.length = 5) :
    ((resultGroups heap (count_values.counts values)).map (ProjectEulerSolutions.TupleSorting.readInts
      (collected heap (count_values.counts values) ++ [.list (resultGroups heap (count_values.counts values))]))).map
        (fun row => row.headD 0) ∈ ProjectEulerSolutions.PokerCountPatterns.profiles := by
  rw [resultGroups_readInts_append heap address (count_values.counts values) found]
  exact resultGroups_profile heap address values found five

/-- The source builds and sorts every frequency group while retaining all caller objects. -/
theorem result (heap : Mutable.Heap) (address : Nat) (table : List (Int × Int)) (globals : Mutable.Globals)
    (found : heap[address]? = some (.dict (count_values.entries table))) :
    ReturnsValue program "rank_groups" [.ref address] heap (.ref (collected heap table).length)
      (collected heap table ++ [.list (resultGroups heap table)]) globals := by
  let entry := functionEnv ((findFunctionIn program.body "rank_groups").getD ([], [])).1
    ((findFunctionIn program.body "rank_groups").getD ([], [])).2 [.ref address] globals
  let ready := setLocal entry "groups" (.ref heap.length)
  have getCounts : ready "counts" = some (.ref address) := by cases globals <;> rfl
  have executed := scan heap address table found table [] (by simp) [] [] ready globals rfl .nil
  simp only [List.nil_append, ProjectEulerSolutions.HeapCollection.working,
    List.length_append, List.length_cons, List.length_nil, Nat.zero_add] at executed
  obtain ⟨finalEnv, getGroups, loopDone, rows⟩ := executed
  apply BlockReturns.function rfl rfl
  apply (BlockContinues.annAssign (Evaluates.list .nil)).returns
  apply (BlockContinues.for_dictionary_items (target := target) (body := sourceBody)
    (object := .dict (count_values.entries table)) (entries := fields table)
    (Evaluates.lookup getCounts) (input_found heap address table found [] []) rfl loopDone).returns
  apply BlockReturns.return_value
  apply Evaluates.builtin (.cons (Evaluates.lookup getGroups) (.cons Evaluates.constant .nil))
  have typed := Represents.typed (collected heap table) (references (heap.length+1) table) table rows
  have ordered := ProjectEulerSolutions.TupleSorting.sortWithReverse (collected heap table) true _ typed
  have collector : (collected heap table)[heap.length]? = some (.list (references (heap.length+1) table)) :=
    ProjectEulerSolutions.HeapCollection.found heap _ _
  exact primitive_sorted_reverse_list (collected heap table) heap.length _ _ true collector ordered

private theorem typed_entries (heap : Mutable.Heap) (address : Nat)
    (typed : Mutable.Value.HasType heap (.ref address) "dict[int,int]") :
    ∃ table : List (Int × Int), heap[address]? = some (.dict (count_values.entries table)) := by
  obtain ⟨entries, found, types⟩ := Mutable.Value.dictionary_of_hasType typed rfl
  have elements : ∀ entry ∈ entries, ∃ key value : Int, entry = (.int key, number value) := by
    intro entry member
    obtain ⟨keyType, scalar, valueEq, valueType⟩ := types entry member
    have keyInt : ∃ key : Int, entry.1 = .int key := by
      cases key : entry.1 <;> simp_all [Mutable.scalarKind, Data.typeName]
    have valueInt : ∃ value : Int, scalar = .int value := by
      cases scalar <;> simp_all [Mutable.scalarKind, Data.typeName]
    obtain ⟨key, keyEq⟩ := keyInt
    obtain ⟨value, valueEq'⟩ := valueInt
    exact ⟨key, value, Prod.ext keyEq (by simpa only [valueEq'] using valueEq)⟩
  have encoded : ∃ table : List (Int × Int), entries = count_values.entries table := by
    clear found types
    induction entries with
    | nil => exact ⟨[], rfl⟩
    | cons first rest ih =>
        obtain ⟨key, value, rfl⟩ := elements first (by simp)
        obtain ⟨table, rfl⟩ := ih (fun entry member => elements entry (List.mem_cons_of_mem _ member))
        exact ⟨(key, value) :: table, rfl⟩
  obtain ⟨table, rfl⟩ := encoded
  exact ⟨table, found⟩

theorem «Returns» (heap : Mutable.Heap) (address : Nat) (_valid : heap.Valid)
    (typed : Mutable.Value.HasType heap (.ref address) "dict[int,int]") :
    (HeapCalls.call program "rank_groups" [.ref address] heap).Returns := by
  obtain ⟨table, found⟩ := typed_entries heap address typed
  exact (result heap address table [] found).returns

theorem «Supported» (heap : Mutable.Heap) (address : Nat) (valid : heap.Valid)
    (typed : Mutable.Value.HasType heap (.ref address) "dict[int,int]") :
    (HeapCalls.call program "rank_groups" [.ref address] heap).Supported := («Returns» heap address valid typed).supported

end rank_groups

namespace parse_card
private theorem first_item (token : String) (rank suit : Char) (rest : List Char)
    (digits : token.toList = rank :: suit :: rest) (heap : Mutable.Heap) :
    Mutable.getItem (.scalar (.string token)) (number 0) heap = some (.ok (character rank), heap) := by
  have inside : ¬ (rest.length : Int)+1+1 ≤ 0 := by omega
  simp [Mutable.getItem, Strings.get, FormalSW.Py.Builtin.Sequence.get?,
    FormalSW.Py.Builtin.Sequence.normalizeIndex, digits, character, inside]

private theorem second_item (token : String) (rank suit : Char) (rest : List Char)
    (digits : token.toList = rank :: suit :: rest) (heap : Mutable.Heap) :
    Mutable.getItem (.scalar (.string token)) (number 1) heap = some (.ok (character suit), heap) := by
  have inside : ¬ (rest.length : Int)+1+1 ≤ 1 := by omega
  simp [Mutable.getItem, Strings.get, FormalSW.Py.Builtin.Sequence.get?,
    FormalSW.Py.Builtin.Sequence.normalizeIndex, digits, character, inside]
/-- The parser reads the supplied rank dictionary and allocates only its returned pair. -/
theorem result (token : String) (rank suit : Char) (rest : List Char) (digits : token.toList = rank :: suit :: rest)
    (heap : Mutable.Heap) (globals : Mutable.Globals) (address : Nat) (entries : List (Constant × Mutable.Value))
    (getTable : globals.lookup "VALUE_MAP" = some (.ref address)) (found : heap[address]? = some (.dict entries))
    (value : Mutable.Value) (present : ScalarDict.lookup entries (.string (String.singleton rank)) = some value) :
    ReturnsValue program "parse_card" [.scalar (.string token)] heap (.ref heap.length)
      (heap ++ [.tuple [value, character suit]]) globals := by
  let entry := functionEnv ((findFunctionIn program.body "parse_card").getD ([], [])).1
    ((findFunctionIn program.body "parse_card").getD ([], [])).2 [.scalar (.string token)] globals
  let ranked := setLocal entry "v" value
  let ready := setLocal ranked "s" (character suit)
  have getToken : entry "token" = some (.scalar (.string token)) := by cases globals <;> rfl
  have table : entry "VALUE_MAP" = some (.ref address) := by
    cases globals with
    | nil => simp at getTable
    | cons pair tail => exact getTable
  have first : Evaluates program entry (.subscript (.name "token") (.constant (.int 0))) heap (character rank) heap globals := by
    apply Evaluates.subscript (middle := heap) (afterIndex := heap)
      (by intro a b c impossible; cases impossible) (Evaluates.lookup getToken) Evaluates.constant
    exact first_item token rank suit rest digits heap
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign (value := value) (middle := heap)
  · apply Evaluates.subscript (middle := heap) (afterIndex := heap)
      (by intro a b c impossible; cases impossible) (Evaluates.lookup table) first
    exact Mutable.getItem_dictionary heap address entries found (.string (String.singleton rank)) present
  · apply BlockReturns.assign (value := character suit) (middle := heap)
    · apply Evaluates.subscript (middle := heap) (afterIndex := heap)
        (by intro a b c impossible; cases impossible) (Evaluates.lookup (by simpa [ranked, setLocal] using getToken)) Evaluates.constant
      exact second_item token rank suit rest digits heap
    · apply BlockReturns.return_value
      exact Evaluates.tuple (.cons (Evaluates.lookup rfl) (.cons (Evaluates.lookup rfl) .nil)) (by simp)
private abbrev context (address : Nat) : Mutable.Globals := [("VALUE_MAP", .ref address)]
private abbrev inputEnv (token : String) (address : Nat) : HeapCalls.Env :=
  fun name => if name = "token" then some (.scalar (.string token)) else (context address).lookup name
private abbrev longEnough (heap : Mutable.Heap) (token : String) (address : Nat) :=
  HeapCalls.Condition program (inputEnv token address) Solvers.P54Ast.requires_parse_card_0 heap (context address)
private abbrev knownRank (heap : Mutable.Heap) (token : String) (address : Nat) :=
  HeapCalls.Condition program (inputEnv token address) Solvers.P54Ast.requires_parse_card_1 heap (context address)

private theorem length_test (heap : Mutable.Heap) (token : String) (address : Nat) :
    Evaluates program (inputEnv token address) Solvers.P54Ast.requires_parse_card_0
      heap (flag (decide (2 ≤ token.toList.length))) heap (context address) := by
  apply Evaluates.compare (av := number token.toList.length) (bv := number 2) (middle := heap) (afterRight := heap)
  · apply Evaluates.builtin (.cons (Evaluates.lookup rfl) .nil)
    simp [primitive, Mutable.length, number]
  · exact Evaluates.constant
  · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
    omega

private theorem token_parts (heap : Mutable.Heap) (token : String) (address : Nat) (enough : longEnough heap token address) :
    ∃ rank suit rest, token.toList = rank :: suit :: rest := by
  have size : 2 ≤ token.toList.length := by simpa [flag] using (length_test heap token address).condition_iff.mp enough
  cases digits : token.toList with
  | nil => simp [digits] at size
  | cons rank tail =>
      cases tail with
      | nil => simp [digits] at size
      | cons suit rest => exact ⟨rank, suit, rest, rfl⟩

private theorem membership_test (heap : Mutable.Heap) (token : String) (address : Nat)
    (rank suit : Char) (rest : List Char) (digits : token.toList = rank :: suit :: rest)
    (entries : List (Constant × Mutable.Value)) (found : heap[address]? = some (.dict entries)) :
    Evaluates program (inputEnv token address) Solvers.P54Ast.requires_parse_card_1 heap
      (flag (ScalarDict.lookup entries (.string (String.singleton rank))).isSome) heap (context address) := by
  apply Evaluates.membership (middle := heap) (needle := character rank) (source := .ref address)
  · refine .cons ?_ (.cons (Evaluates.lookup rfl) .nil)
    exact Evaluates.subscript (by intro a b c impossible; cases impossible) (Evaluates.lookup rfl) Evaluates.constant
      (first_item token rank suit rest digits heap)
  · exact membership_dictionary (.string (String.singleton rank)) false found

theorem «Returns» (heap : Mutable.Heap) (token : String) (address : Nat)
    (_valid : heap.Valid) (typed : Mutable.Value.HasType heap (.ref address) "dict[str,int]")
    (enough : longEnough heap token address) (known : knownRank heap token address) :
    (HeapCalls.call program "parse_card" [.scalar (.string token)] heap (context address)).Returns := by
  obtain ⟨rank, suit, rest, digits⟩ := token_parts heap token address enough
  obtain ⟨entries, found, _⟩ := Mutable.Value.dictionary_of_hasType typed rfl
  have present : (ScalarDict.lookup entries (.string (String.singleton rank))).isSome = true := by
    simpa [flag] using (membership_test heap token address rank suit rest digits entries found).condition_iff.mp known
  cases selected : ScalarDict.lookup entries (.string (String.singleton rank)) with
  | none => simp [selected] at present
  | some value => exact (result token rank suit rest digits heap (context address) address entries rfl found value selected).returns

theorem «Supported» (heap : Mutable.Heap) (token : String) (address : Nat)
    (valid : heap.Valid) (typed : Mutable.Value.HasType heap (.ref address) "dict[str,int]")
    (enough : longEnough heap token address) (known : knownRank heap token address) :
    (HeapCalls.call program "parse_card" [.scalar (.string token)] heap (context address)).Supported :=
  («Returns» heap token address valid typed enough known).supported

end parse_card
theorem requires_parse_card_0_supported (heap : Mutable.Heap) (token : String) (address : Nat)
    (_valid : heap.Valid) (_typed : Mutable.Value.HasType heap (.ref address) "dict[str,int]") :
    (HeapCalls.evaluate program (parse_card.inputEnv token address) Solvers.P54Ast.requires_parse_card_0 heap (parse_card.context address)).Supported :=
  Execution.supported_of_finishes (parse_card.length_test heap token address).finishes

theorem requires_parse_card_1_supported (heap : Mutable.Heap) (token : String) (address : Nat)
    (_valid : heap.Valid) (typed : Mutable.Value.HasType heap (.ref address) "dict[str,int]")
    (enough : parse_card.longEnough heap token address) :
    (HeapCalls.evaluate program (parse_card.inputEnv token address) Solvers.P54Ast.requires_parse_card_1 heap (parse_card.context address)).Supported := by
  obtain ⟨rank, suit, rest, digits⟩ := parse_card.token_parts heap token address enough
  obtain ⟨entries, found, _⟩ := Mutable.Value.dictionary_of_hasType typed rfl
  exact Execution.supported_of_finishes (parse_card.membership_test heap token address rank suit rest digits entries found).finishes

namespace player1_wins

/-- Tokens are separated by Python's Unicode whitespace rules. -/
def tokens (line : String) : List String := Strings.splitWhitespace (Strings.stripWhitespace line)

private abbrev text (value : String) : Mutable.Value := .scalar (.string value)
private abbrev values (line : String) : List Mutable.Value := (tokens line).map text
private abbrev splitHeap (heap : Mutable.Heap) (line : String) : Mutable.Heap := heap ++ [.list (values line)]

/-- Tokenization evaluates both string methods and allocates exactly one list. -/
theorem tokenize (line : String) (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getLine : env "line" = some (text line)) :
    Evaluates program env (.call (.attribute (.call (.attribute (.name "line") "strip") []) "split") [])
      heap (.ref heap.length) (splitHeap heap line) globals := by
  apply Evaluates.split_whitespace
  · exact Evaluates.strip (Evaluates.lookup getLine) (by intro name equal; cases equal; simp [getLine])
  · intro name impossible
    cases impossible

private theorem empty_test (line : String) (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getParts : env "parts" = some (.ref heap.length)) :
    Evaluates program env (.unaryOp .not (.name "parts")) (splitHeap heap line)
      (flag (tokens line).isEmpty) (splitHeap heap line) globals := by
  apply Evaluates.unary (Evaluates.lookup getParts)
  simp [HeapCalls.unary, Mutable.truth, splitHeap, values, flag]

/-- Empty and whitespace-only lines return false without calling either hand evaluator. -/
theorem blank (line : String) (heap : Mutable.Heap) (globals : Mutable.Globals) (empty : tokens line = []) :
    ReturnsValue program "player1_wins" [text line] heap (flag false) (heap ++ [.list []]) globals := by
  let entry := functionEnv ((findFunctionIn program.body "player1_wins").getD ([], [])).1
    ((findFunctionIn program.body "player1_wins").getD ([], [])).2 [text line] globals
  let ready := setLocal entry "parts" (.ref heap.length)
  have getLine : entry "line" = some (text line) := by cases globals <;> rfl
  have allocated : splitHeap heap line = heap ++ [.list []] := by simp [splitHeap, values, empty]
  rw [← allocated]
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign (tokenize line heap entry globals getLine)
  apply BlockReturns.branch (flag := true) (value := flag true) (middle := splitHeap heap line)
  · simpa [empty] using empty_test line heap ready globals rfl
  · rfl
  · exact BlockReturns.return_value Evaluates.constant

/-- Every all-whitespace Unicode line takes the empty-line branch. -/
theorem whitespace (line : String) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (onlyWhitespace : line.toList.all Strings.isWhitespace = true) :
    ReturnsValue program "player1_wins" [text line] heap (flag false) (heap ++ [.list []]) globals := by
  have dropAll : ∀ characters : List Char, characters.all Strings.isWhitespace = true →
      characters.dropWhile Strings.isWhitespace = [] := by
    intro characters
    induction characters with
    | nil => intro _; rfl
    | cons character rest ih =>
        intro all
        have both := Bool.and_eq_true_iff.mp all
        rw [List.dropWhile_cons_of_pos both.1]
        exact ih both.2
  apply blank line heap globals
  have stripped : Strings.stripWhitespace line = "" := by
    simp [Strings.stripWhitespace, dropAll line.toList onlyWhitespace]
  rw [tokens, stripped]
  rfl

/-- The two slices are separate fresh lists, even for lines containing fewer than ten tokens. -/
def handsHeap (heap : Mutable.Heap) (line : String) : Mutable.Heap :=
  splitHeap heap line ++ [.list ((values line).take 5), .list ((values line).drop 5)]

private theorem first_hand (line : String) (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getParts : env "parts" = some (.ref heap.length)) :
    Evaluates program env (.subscript (.name "parts") (.slice none (some (.constant (.int 5))) none))
      (splitHeap heap line) (.ref (heap.length+1)) (splitHeap heap line ++ [.list ((values line).take 5)]) globals := by
  apply Evaluates.slice (.cons (Evaluates.lookup getParts) (.cons Evaluates.constant
    (.cons Evaluates.constant (.cons Evaluates.constant .nil))))
  have sliced := Mutable.getSlice_list_prefix (splitHeap heap line) heap.length (values line) 5 (by simp [splitHeap])
  have prefixValues : (values line).take (FormalSW.Py.Builtin.Slice.prefixLength (values line).length 5) =
      (values line).take 5 := by
    rw [show (5 : Int) = ((5 : Nat) : Int) from rfl, FormalSW.Py.Builtin.Slice.prefixLength_nonnegative]
    by_cases short : (values line).length ≤ 5
    · rw [Nat.min_eq_right short, List.take_length, List.take_of_length_le short]
    · rw [Nat.min_eq_left (by omega)]
  simpa only [prefixValues, splitHeap, List.length_append, List.length_singleton] using sliced

private theorem second_hand (line : String) (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getParts : env "parts" = some (.ref heap.length)) :
    Evaluates program env (.subscript (.name "parts") (.slice (some (.constant (.int 5))) none none))
      (splitHeap heap line ++ [.list ((values line).take 5)]) (.ref (heap.length+2)) (handsHeap heap line) globals := by
  apply Evaluates.slice (.cons (Evaluates.lookup getParts) (.cons Evaluates.constant
    (.cons Evaluates.constant (.cons Evaluates.constant .nil))))
  have retained : (splitHeap heap line ++ [.list ((values line).take 5)])[heap.length]? = some (.list (values line)) := by
    simp [splitHeap, List.append_assoc]
  have sliced := Mutable.getSlice_list_suffix (splitHeap heap line ++ [.list ((values line).take 5)]) heap.length (values line) 5 retained
  have size : (splitHeap heap line ++ [Mutable.Object.list ((values line).take 5)]).length = heap.length+2 := by
    simp only [splitHeap, List.length_append, List.length_singleton, Nat.add_assoc]
  have merged : splitHeap heap line ++ [.list ((values line).take 5)] ++ [.list ((values line).drop 5)] = handsHeap heap line := by
    rw [List.append_assoc]
    rfl
  rw [size, merged] at sliced
  exact sliced

/-- Once the rank calls and their comparison are proved, this composes the complete nonempty-line path. -/
theorem ranked (line : String) (heap : Mutable.Heap) (globals : Mutable.Globals) (nonempty : tokens line ≠ [])
    (left right : Mutable.Value) (middle finalHeap : Mutable.Heap) (wins : Bool)
    (first : ReturnsValue program "hand_rank" [.ref (heap.length+1)] (handsHeap heap line) left middle globals)
    (second : ReturnsValue program "hand_rank" [.ref (heap.length+2)] middle right finalHeap globals)
    (order : HeapCalls.compare .gt left right finalHeap = some (.ok (flag wins), finalHeap)) :
    ReturnsValue program "player1_wins" [text line] heap (flag wins) finalHeap globals := by
  let entry := functionEnv ((findFunctionIn program.body "player1_wins").getD ([], [])).1
    ((findFunctionIn program.body "player1_wins").getD ([], [])).2 [text line] globals
  let parsed := setLocal entry "parts" (.ref heap.length)
  let dealt := setLocal (setLocal parsed "h1" (.ref (heap.length+1))) "h2" (.ref (heap.length+2))
  have getLine : entry "line" = some (text line) := by cases globals <;> rfl
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign (tokenize line heap entry globals getLine)
  apply BlockReturns.branch (flag := false) (value := flag false) (middle := splitHeap heap line)
  · simpa [List.isEmpty_eq_false_iff.mpr nonempty] using empty_test line heap parsed globals rfl
  · rfl
  · apply BlockReturns.assign (first_hand line heap parsed globals rfl)
    apply BlockReturns.assign (second_hand line heap (setLocal parsed "h1" (.ref (heap.length+1))) globals (by simp [setLocal, parsed]))
    apply BlockReturns.return_value
    exact Evaluates.compare (env := dealt)
      (Evaluates.call (.cons (Evaluates.lookup (by simp [dealt, setLocal])) .nil) first)
      (Evaluates.call (.cons (Evaluates.lookup rfl) .nil) second) order

/-- Different hand categories decide the winner without inspecting the tie-breaking fields. -/
theorem different_categories (line : String) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (nonempty : tokens line ≠ []) (a b : Nat) (firstCategory secondCategory : Int)
    (leftKeys rightKeys : List Mutable.Value) (middle finalHeap : Mutable.Heap)
    (first : ReturnsValue program "hand_rank" [.ref (heap.length+1)] (handsHeap heap line) (.ref a) middle globals)
    (second : ReturnsValue program "hand_rank" [.ref (heap.length+2)] middle (.ref b) finalHeap globals)
    (left : finalHeap[a]? = some (.tuple (number firstCategory :: leftKeys)))
    (right : finalHeap[b]? = some (.tuple (number secondCategory :: rightKeys)))
    (different : firstCategory ≠ secondCategory) :
    ReturnsValue program "player1_wins" [.scalar (.string line)] heap
      (flag (decide (secondCategory < firstCategory))) finalHeap globals := by
  exact ranked line heap globals nonempty (.ref a) (.ref b) middle finalHeap _ first second
    (compare_tuple_head_gt finalHeap a b firstCategory secondCategory leftKeys rightKeys left right different)

end player1_wins
end solvers.«54»
