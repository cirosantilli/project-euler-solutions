import «.formalsw».solvers.«54.py.ast»
import lang.py.FormalSW.DictionaryCalls
import lang.py.FormalSW.DictionaryItemsCalls
import lang.py.FormalSW.ReverseSortingCalls
import lang.py.FormalSW.TupleComparisons
import ProjectEulerSolutions.Lib.HeapCollection
import ProjectEulerSolutions.Lib.PokerGroupOrder
import ProjectEulerSolutions.Lib.PokerUnique
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

namespace solvers.«54».hand_rank
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
private abbrev program := Solvers.P54Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
private abbrev text (value : String) : Mutable.Value := .scalar (.string value)
private abbrev character (value : Char) : Mutable.Value := text (String.singleton value)

/-- Tokens retain arbitrary suffixes; parsing reads their first two Unicode characters. -/
structure Card where
  token : String
  rankChar : Char
  suitChar : Char
  suffix : List Char
  value : Int
  shape : token.toList = rankChar :: suitChar :: suffix

def tokens (cards : List Card) : List Mutable.Value := cards.map fun card => text card.token
def values (cards : List Card) : List Mutable.Value := cards.map fun card => number card.value
def suits (cards : List Card) : List Mutable.Value := cards.map fun card => character card.suitChar
def pairs (cards : List Card) : Mutable.Heap := cards.map fun card => .tuple [number card.value, character card.suitChar]

/-- The parser owns two buffers followed by the actual returned parse-card tuples. -/
def parsedHeap (heap : Mutable.Heap) (cards : List Card) : Mutable.Heap :=
  heap ++ [.list (values cards), .list (suits cards)] ++ pairs cards

theorem values_found (heap : Mutable.Heap) (cards : List Card) :
    (parsedHeap heap cards)[heap.length]? = some (.list (values cards)) := by
  simp [parsedHeap, List.append_assoc]

theorem suits_found (heap : Mutable.Heap) (cards : List Card) :
    (parsedHeap heap cards)[heap.length+1]? = some (.list (suits cards)) := by
  simp [parsedHeap, List.append_assoc]

theorem caller_found (heap : Mutable.Heap) (cards : List Card) (address : Nat) (object : Mutable.Object)
    (found : heap[address]? = some object) : (parsedHeap heap cards)[address]? = some object := by
  simpa only [parsedHeap, List.append_assoc] using
    ProjectEulerSolutions.HeapCollection.prefix_found heap
      ([.list (values cards), .list (suits cards)] ++ pairs cards) address object found

private def cardPair (card : Card) : Mutable.Object := .tuple [number card.value, character card.suitChar]

private def afterValues (heap : Mutable.Heap) (before : List Card) (card : Card) : Mutable.Heap :=
  (parsedHeap heap before ++ [cardPair card]).set heap.length (.list (values before ++ [number card.value]))

private theorem stored (heap : Mutable.Heap) (before : List Card) (card : Card) :
    (afterValues heap before card).set (heap.length+1) (.list (suits before ++ [character card.suitChar])) =
      parsedHeap heap (before ++ [card]) := by
  simp [afterValues, parsedHeap, values, suits, pairs, cardPair, List.map_append,
    List.append_assoc, List.set_append_right]

private def body : List Stmt :=
  [.assign [.tuple [.name "v", .name "s"]] (.call (.name "parse_card") [.name "c"]),
   .expr (.call (.attribute (.name "values") "append") [.name "v"]),
   .expr (.call (.attribute (.name "suits") "append") [.name "s"])]

private theorem one_card (heap : Mutable.Heap) (before : List Card) (card : Card)
    (globals : Mutable.Globals) (tableAddress : Nat) (entries : List (Constant × Mutable.Value))
    (getTable : globals.lookup "VALUE_MAP" = some (.ref tableAddress))
    (tableFound : heap[tableAddress]? = some (.dict entries))
    (rankFound : ScalarDict.lookup entries (.string (String.singleton card.rankChar)) = some (number card.value))
    (env : HeapCalls.Env) (getCard : env "c" = some (text card.token))
    (getValues : env "values" = some (.ref heap.length)) (getSuits : env "suits" = some (.ref (heap.length+1))) :
    BlockContinues program env body (parsedHeap heap before)
      (assignNames env ["v", "s"] [number card.value, character card.suitChar])
      (parsedHeap heap (before ++ [card])) globals := by
  let current := parsedHeap heap before
  let ready := assignNames env ["v", "s"] [number card.value, character card.suitChar]
  have parsed := solvers.«54».parse_card.result card.token card.rankChar card.suitChar card.suffix card.shape
    current globals tableAddress entries getTable (caller_found heap before tableAddress _ tableFound)
    (number card.value) rankFound
  have called : Evaluates program env (.call (.name "parse_card") [.name "c"]) current (.ref current.length)
      (current ++ [cardPair card]) globals :=
    Evaluates.call (.cons (Evaluates.lookup getCard) .nil) parsed
  have unpacked := BlockContinues.unpack (module := program) (env := env) (globals := globals)
    (heap := current) (value := .ref current.length) (finalHeap := current ++ [cardPair card])
    (expressions := [.name "v", .name "s"]) (names := ["v", "s"])
    (items := [number card.value, character card.suitChar])
    rfl called (by simp [Mutable.sequence, cardPair]) rfl
  have getV : ready "v" = some (number card.value) := by simp [ready, assignNames, setLocal]
  have getS : ready "s" = some (character card.suitChar) := by simp [ready, assignNames, setLocal]
  have getReadyValues : ready "values" = some (.ref heap.length) := by
    simpa [ready, assignNames, setLocal] using getValues
  have getReadySuits : ready "suits" = some (.ref (heap.length+1)) := by
    simpa [ready, assignNames, setLocal] using getSuits
  have valuesAtPair : (current ++ [cardPair card])[heap.length]? = some (.list (values before)) :=
    ProjectEulerSolutions.HeapCollection.prefix_found current [cardPair card] heap.length _ (values_found heap before)
  have suitsAtPair : (current ++ [cardPair card])[heap.length+1]? = some (.list (suits before)) :=
    ProjectEulerSolutions.HeapCollection.prefix_found current [cardPair card] (heap.length+1) _ (suits_found heap before)
  have appendedValues : BlockContinues program ready
      [.expr (.call (.attribute (.name "values") "append") [.name "v"])]
      (current ++ [cardPair card]) ready (afterValues heap before card) globals :=
    BlockContinues.discard (Evaluates.append (Evaluates.lookup getReadyValues) valuesAtPair
      (Evaluates.lookup getV) valuesAtPair)
  have suitsAfterValues : (afterValues heap before card)[heap.length+1]? = some (.list (suits before)) := by
    unfold afterValues
    rw [List.getElem?_set_ne (by omega : heap.length ≠ heap.length+1)]
    exact suitsAtPair
  have appendedSuits := BlockContinues.discard (Evaluates.append (module := program) (env := ready) (globals := globals)
    (Evaluates.lookup getReadySuits) suitsAfterValues (Evaluates.lookup getS) suitsAfterValues)
  apply BlockContinues.append unpacked
  apply BlockContinues.append appendedValues
  simpa only [stored] using appendedSuits

/-- Every record is interpreted through the caller's live dictionary. -/
def Ranked (entries : List (Constant × Mutable.Value)) (cards : List Card) : Prop :=
  ∀ card ∈ cards, ScalarDict.lookup entries (.string (String.singleton card.rankChar)) = some (number card.value)

/-- An arbitrary already-parsed portion composes with all remaining source iterations. -/
theorem parse_remaining (heap : Mutable.Heap) (cardsAddress : Nat) (cards : List Card)
    (cardsFound : heap[cardsAddress]? = some (.list (tokens cards)))
    (globals : Mutable.Globals) (tableAddress : Nat) (entries : List (Constant × Mutable.Value))
    (getTable : globals.lookup "VALUE_MAP" = some (.ref tableAddress))
    (tableFound : heap[tableAddress]? = some (.dict entries)) (ranked : Ranked entries cards)
    (remaining before : List Card) (decomposition : cards = before ++ remaining)
    (env : HeapCalls.Env) (getValues : env "values" = some (.ref heap.length))
    (getSuits : env "suits" = some (.ref (heap.length+1))) :
    ∃ finalEnv, finalEnv "values" = some (.ref heap.length) ∧ finalEnv "suits" = some (.ref (heap.length+1)) ∧
      BlockContinues program env [.sequenceStepLoop "c" (.ref cardsAddress) before.length body []]
        (parsedHeap heap before) finalEnv (parsedHeap heap cards) globals := by
  induction remaining generalizing before env with
  | nil =>
      refine ⟨env, getValues, getSuits, ?_⟩
      apply BlockContinues.sequence_done (items := tokens cards)
      · simp [sequenceValue, Mutable.sequence, caller_found heap before cardsAddress _ cardsFound]
      · simp [tokens, decomposition]
      · simpa only [decomposition, List.append_nil] using
          (BlockContinues.nil (module := program) (env := env) (heap := parsedHeap heap before) (globals := globals))
  | cons card rest ih =>
      let bound := setLocal env "c" (text card.token)
      let ready := assignNames bound ["v", "s"] [number card.value, character card.suitChar]
      have known : ScalarDict.lookup entries (.string (String.singleton card.rankChar)) = some (number card.value) :=
        ranked card (by simp [decomposition])
      have nextValues : ready "values" = some (.ref heap.length) := by
        simpa [ready, bound, assignNames, setLocal] using getValues
      have nextSuits : ready "suits" = some (.ref (heap.length+1)) := by
        simpa [ready, bound, assignNames, setLocal] using getSuits
      have nextDecomposition : cards = (before ++ [card]) ++ rest := by
        simpa only [List.append_assoc, List.singleton_append] using decomposition
      obtain ⟨finalEnv, finalValues, finalSuits, continued⟩ :=
        ih (before ++ [card]) nextDecomposition ready nextValues nextSuits
      refine ⟨finalEnv, finalValues, finalSuits, ?_⟩
      apply BlockContinues.sequence_step (items := tokens cards) (value := text card.token)
      · simp [sequenceValue, Mutable.sequence, caller_found heap before cardsAddress _ cardsFound]
      · simp [tokens, decomposition]
      · apply BlockContinues.append (one_card heap before card globals tableAddress entries getTable tableFound known bound rfl
          (by simpa [bound, setLocal] using getValues) (by simpa [bound, setLocal] using getSuits))
        change BlockContinues program ready
          [.sequenceStepLoop "c" (.ref cardsAddress) (before.length+1) body []]
          (parsedHeap heap (before ++ [card])) finalEnv (parsedHeap heap cards) globals
        simpa only [List.length_append, List.length_singleton] using continued

/-- The source loop fills both fresh buffers, preserving the whole caller heap. -/
theorem parse_loop (heap : Mutable.Heap) (cardsAddress : Nat) (cards : List Card)
    (cardsFound : heap[cardsAddress]? = some (.list (tokens cards)))
    (globals : Mutable.Globals) (tableAddress : Nat) (entries : List (Constant × Mutable.Value))
    (getTable : globals.lookup "VALUE_MAP" = some (.ref tableAddress))
    (tableFound : heap[tableAddress]? = some (.dict entries)) (ranked : Ranked entries cards)
    (env : HeapCalls.Env) (getCards : env "cards" = some (.ref cardsAddress))
    (getValues : env "values" = some (.ref heap.length)) (getSuits : env "suits" = some (.ref (heap.length+1))) :
    ∃ finalEnv, finalEnv "values" = some (.ref heap.length) ∧ finalEnv "suits" = some (.ref (heap.length+1)) ∧
      BlockContinues program env [.forIn (.name "c") (.name "cards") body []]
        (parsedHeap heap []) finalEnv (parsedHeap heap cards) globals := by
  obtain ⟨finalEnv, finalValues, finalSuits, continued⟩ :=
    parse_remaining heap cardsAddress cards cardsFound globals tableAddress entries getTable tableFound ranked
      cards [] (by simp) env getValues getSuits
  refine ⟨finalEnv, finalValues, finalSuits, ?_⟩
  apply BlockContinues.for_sequence (items := tokens cards)
    (by intro expressions impossible; cases impossible) (Evaluates.lookup getCards)
  · have preserved : (parsedHeap heap [])[cardsAddress]? = some (.list (tokens cards)) :=
      caller_found heap [] cardsAddress _ cardsFound
    simp [Mutable.sequence, preserved]
  · exact continued

/-- The parsing wrapper composes with the numeric scorer's actual source execution. -/
theorem result (heap : Mutable.Heap) (cardsAddress : Nat) (cards : List Card)
    (cardsFound : heap[cardsAddress]? = some (.list (tokens cards)))
    (globals : Mutable.Globals) (tableAddress : Nat) (entries : List (Constant × Mutable.Value))
    (getTable : globals.lookup "VALUE_MAP" = some (.ref tableAddress))
    (tableFound : heap[tableAddress]? = some (.dict entries)) (ranked : Ranked entries cards)
    (answer : Mutable.Value) (finalHeap : Mutable.Heap)
    (scored : ReturnsValue program "_score_values" [.ref heap.length, .ref (heap.length+1)]
      (parsedHeap heap cards) answer finalHeap globals) :
    ReturnsValue program "hand_rank" [.ref cardsAddress] heap answer finalHeap globals := by
  let entry := functionEnv ((findFunctionIn program.body "hand_rank").getD ([], [])).1
    ((findFunctionIn program.body "hand_rank").getD ([], [])).2 [.ref cardsAddress] globals
  let valuesEnv := setLocal entry "values" (.ref heap.length)
  let ready := setLocal valuesEnv "suits" (.ref (heap.length+1))
  have getCards : ready "cards" = some (.ref cardsAddress) := by cases globals <;> rfl
  have suitsAllocated : Evaluates program valuesEnv (.list []) (heap ++ [.list []]) (.ref (heap.length+1))
      (parsedHeap heap []) globals := by
    have allocated : Evaluates program valuesEnv (.list []) (heap ++ [.list []])
        (.ref (heap ++ [Mutable.Object.list []]).length)
        ((heap ++ [.list []]) ++ [.list []]) globals := Evaluates.list .nil
    simpa [parsedHeap, values, suits, pairs, List.append_assoc] using allocated
  obtain ⟨finalEnv, finalValues, finalSuits, parsed⟩ :=
    parse_loop heap cardsAddress cards cardsFound globals tableAddress entries getTable tableFound ranked ready getCards
      (by simp [ready, setLocal, valuesEnv]) rfl
  apply BlockReturns.function rfl rfl
  apply (BlockContinues.annAssign (Evaluates.list .nil)).returns
  apply (BlockContinues.annAssign suitsAllocated).returns
  apply parsed.returns
  exact BlockReturns.return_value (Evaluates.call
    (.cons (Evaluates.lookup finalValues) (.cons (Evaluates.lookup finalSuits) .nil)) scored)

end solvers.«54».hand_rank

/-! Operational phases of the numeric poker scorer. -/
namespace solvers.«54».numeric_score
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
set_option Elab.async false

private abbrev program := Solvers.P54Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
private abbrev numbers (values : List Int) : List Mutable.Value := values.map number
private abbrev texts (values : List String) : List Mutable.Value := values.map fun value => .scalar (.string value)
private abbrev flag (value : Bool) : Mutable.Value := .scalar (.bool value)
private def code : List Stmt := executableBody ((findFunctionIn program.body "_score_values").getD ([], [])).2

def addUnique (values : List Int) (value : Int) : List Int :=
  if value ∈ values then values else values ++ [value]

def unique (values : List Int) : List Int := values.foldl addUnique []
def descending (values : List Int) : List Int := HeapCalls.sortedInts true values
def orderedUnique (values : List Int) : List Int := ScalarSort.ints (unique (descending values))
def suitKeys (suits : List String) : List Constant := (suits.map Constant.string).foldl ScalarSet.insert []
def isFlush (suits : List String) : Bool := decide ((suitKeys suits).length = 1)

private theorem fold_ints (remaining done : List Int) :
    (remaining.map Constant.int).foldl ScalarSet.insert (done.map Constant.int) =
      (remaining.foldl addUnique done).map Constant.int := by
  induction remaining generalizing done with
  | nil => rfl
  | cons first rest ih =>
      change (rest.map Constant.int).foldl ScalarSet.insert (ScalarSet.insert (done.map Constant.int) (.int first)) =
        (rest.foldl addUnique (addUnique done first)).map Constant.int
      rw [ScalarSet.insert_ints]
      exact ih (addUnique done first)

private theorem unique_keys (values : List Int) :
    (values.map Constant.int).foldl ScalarSet.insert [] = (unique values).map Constant.int :=
  fold_ints values []

private theorem found_append {heap : Heap} {address : Nat} {object : Mutable.Object}
    (found : heap[address]? = some object) (extra : Heap) : (heap ++ extra)[address]? = some object := by
  rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1]
  exact found

theorem buffers_distinct (heap : Heap) (valuesAddress suitsAddress : Nat) (values : List Int) (suits : List String)
    (nonempty : values ≠ [])
    (valuesFound : heap[valuesAddress]? = some (.list (numbers values)))
    (suitsFound : heap[suitsAddress]? = some (.list (texts suits))) : valuesAddress ≠ suitsAddress := by
  intro same
  subst suitsAddress
  have objects := Option.some.inj (valuesFound.symm.trans suitsFound)
  cases values with
  | nil => exact nonempty rfl
  | cons value rest => cases suits <;> simp [numbers, number, texts] at objects

private theorem set_texts (heap : Heap) (env : HeapCalls.Env) (address : Nat) (suits : List String) (globals : Mutable.Globals)
    (bound : env "suits" = some (.ref address)) (found : heap[address]? = some (.list (texts suits))) :
    Evaluates program env (.builtinCall .set [.name "suits"]) heap (.ref heap.length)
      (heap ++ [.set (suitKeys suits)]) globals := by
  apply Evaluates.builtin (.cons (Evaluates.lookup bound) .nil)
  simpa [primitive, suitKeys, texts, List.map_map, Function.comp_def] using
    Mutable.setFrom_list heap address (suits.map Constant.string)
      (by simpa [texts, List.map_map, Function.comp_def] using found)

private theorem flush_expression (heap : Heap) (env : HeapCalls.Env) (address : Nat) (suits : List String) (globals : Mutable.Globals)
    (bound : env "suits" = some (.ref address)) (found : heap[address]? = some (.list (texts suits))) :
    Evaluates program env (.compare (.builtinCall .len [.builtinCall .set [.name "suits"]]) [(.eq, .constant (.int 1))])
      heap (flag (isFlush suits)) (heap ++ [.set (suitKeys suits)]) globals := by
  let middle := heap ++ [Mutable.Object.set (suitKeys suits)]
  apply Evaluates.compare (av := number (suitKeys suits).length) (bv := number 1) (middle := middle) (afterRight := middle)
  · apply Evaluates.builtin (.cons (set_texts heap env address suits globals bound found) .nil)
    simp [primitive, Mutable.length, middle, number]
  · exact Evaluates.constant
  · simp [HeapCalls.compare, scalarCompare, evalCmpOp, isFlush, flag]
    constructor
    · omega
    · rfl

private theorem set_ints (heap : Heap) (env : HeapCalls.Env) (address : Nat) (values : List Int) (globals : Mutable.Globals)
    (bound : env "values" = some (.ref address)) (found : heap[address]? = some (.list (numbers values))) :
    Evaluates program env (.builtinCall .set [.name "values"]) heap (.ref heap.length)
      (heap ++ [.set ((unique values).map Constant.int)]) globals := by
  apply Evaluates.builtin (.cons (Evaluates.lookup bound) .nil)
  simpa [primitive, unique_keys, numbers, number, List.map_map, Function.comp_def] using
    Mutable.setFrom_list heap address (values.map Constant.int)
      (by simpa [numbers, number, List.map_map, Function.comp_def] using found)

private theorem unique_expression (heap : Heap) (env : HeapCalls.Env) (address : Nat) (values : List Int) (globals : Mutable.Globals)
    (bound : env "values" = some (.ref address)) (found : heap[address]? = some (.list (numbers values))) :
    Evaluates program env (.builtinCall .sorted [.builtinCall .set [.name "values"]]) heap (.ref (heap.length+1))
      (heap ++ [.set ((unique values).map Constant.int), .list (numbers (ScalarSort.ints (unique values)))]) globals := by
  apply Evaluates.builtin (.cons (set_ints heap env address values globals bound found) .nil)
  simp [primitive, HeapSort.sort_scalars, ScalarSort.sort_ints, Mutable.allocate,
    Except.map, numbers, List.map_map, Function.comp_def, List.append_assoc]

def normalizationHeap (heap : Heap) (address : Nat) (values : List Int) (suits : List String) : Heap :=
  heap.set address (.list (numbers (descending values))) ++
    [.set (suitKeys suits), .set ((unique (descending values)).map Constant.int), .list (numbers (orderedUnique values))]

def normalizationEnv (env : HeapCalls.Env) (heap : Heap) (suits : List String) : HeapCalls.Env :=
  setLocal (setLocal (setLocal (setLocal env "is_flush" (flag (isFlush suits)))
    "uniq" (.ref (heap.length+2))) "is_straight" (flag false)) "straight_high" (number 0)

/-- The first five source statements sort the values buffer and allocate both sets plus the sorted unique list. -/
theorem normalization (heap : Heap) (env : HeapCalls.Env) (valuesAddress suitsAddress : Nat)
    (values : List Int) (suits : List String) (globals : Mutable.Globals)
    (nonempty : values ≠ [])
    (valuesFound : heap[valuesAddress]? = some (.list (numbers values)))
    (suitsFound : heap[suitsAddress]? = some (.list (texts suits)))
    (getValues : env "values" = some (.ref valuesAddress)) (getSuits : env "suits" = some (.ref suitsAddress)) :
    BlockContinues program env (code.take 5) heap (normalizationEnv env heap suits)
      (normalizationHeap heap valuesAddress values suits) globals := by
  let afterSort := heap.set valuesAddress (Mutable.Object.list (numbers (descending values)))
  let afterSuits := afterSort ++ [Mutable.Object.set (suitKeys suits)]
  let flushEnv := setLocal env "is_flush" (flag (isFlush suits))
  let uniqEnv := setLocal flushEnv "uniq" (.ref (afterSuits.length+1))
  let straightEnv := setLocal uniqEnv "is_straight" (flag false)
  have inside := (List.getElem?_eq_some_iff.mp valuesFound).1
  have different := buffers_distinct heap valuesAddress suitsAddress values suits nonempty valuesFound suitsFound
  have sortedFound : afterSort[valuesAddress]? = some (.list (numbers (descending values))) := by simp [afterSort, inside]
  have suitsPreserved : afterSort[suitsAddress]? = some (.list (texts suits)) := by
    rw [List.getElem?_set_ne different]
    exact suitsFound
  have valuesPreserved : afterSuits[valuesAddress]? = some (.list (numbers (descending values))) :=
    found_append sortedFound _
  have sorted : Evaluates program env (.call (.attribute (.name "values") "$formalsw.sort.reverse") [.constant (.bool true)])
      heap (.scalar .none) afterSort globals := by
    apply Evaluates.sort_reverse (Evaluates.lookup getValues) valuesFound Evaluates.constant
    · simpa [numbers, number, descending, afterSort, List.map_map, Function.comp_def] using
        primitive_sort_reverse_ints heap valuesAddress values true
          (by simpa [numbers, number, List.map_map, Function.comp_def] using valuesFound)
    · intro name same
      cases same
      simp [getValues]
  have flushed := BlockContinues.assign (name := "is_flush")
    (flush_expression afterSort env suitsAddress suits globals getSuits suitsPreserved)
  have uniques := BlockContinues.assign (name := "uniq")
    (unique_expression afterSuits flushEnv valuesAddress (descending values) globals
      (by simpa [flushEnv, setLocal] using getValues) valuesPreserved)
  have first := (BlockContinues.discard sorted).append (flushed.append (uniques.append
    ((BlockContinues.assign (name := "is_straight") (env := uniqEnv)
      (Evaluates.constant (value := .bool false))).append
      (BlockContinues.assign (name := "straight_high") (env := straightEnv)
        (Evaluates.constant (value := .int 0))))))
  simpa [code, normalizationEnv, normalizationHeap, afterSort, afterSuits, flushEnv, uniqEnv,
    straightEnv, orderedUnique, List.append_assoc, Nat.add_assoc] using first

end solvers.«54».numeric_score

namespace solvers.«54».Kickers
open FormalSW.PyAst

/-- A rank and all the other ranks partition the original list. -/
theorem length_partition (values : List Int) (rank : Int) :
    values.length = values.count rank + (values.filter (fun value => value != rank)).length := by
  have partition := List.length_eq_length_filter_add (l := values) (fun value => value == rank)
  rw [← List.count_eq_length_filter] at partition
  exact partition

theorem four_kicker_nonempty (values : List Int) (four : Int)
    (five : values.length = 5) (counted : values.count four = 4) :
    values.filter (fun value => value != four) ≠ [] := by
  intro empty
  have partition := length_partition values four
  rw [empty] at partition
  simp only [List.length_nil] at partition
  omega

theorem two_pairs_kicker_nonempty (values : List Int) (high low : Int)
    (five : values.length = 5) (different : high ≠ low)
    (highCount : values.count high = 2) (lowCount : values.count low = 2) :
    values.filter (fun value => value != high && value != low) ≠ [] := by
  intro empty
  have first := length_partition values high
  have second := length_partition (values.filter (fun value => value != high)) low
  have lowPreserved : (values.filter (fun value => value != high)).count low = values.count low :=
    List.count_filter (by simp [Ne.symm different])
  have combined : (values.filter (fun value => value != high)).filter (fun value => value != low) =
      values.filter (fun value => value != high && value != low) := by
    simp only [List.filter_filter, Bool.and_comm]
  rw [lowPreserved, combined, empty] at second
  simp only [List.length_nil] at second
  omega

/-- Unique dictionary keys identify the value selected by the checked lookup function. -/
theorem lookup_of_mem (table : List (Int × Int)) (entry : Int × Int)
    (member : entry ∈ table) (unique : (table.map Prod.fst).Nodup) :
    count_values.lookup table entry.1 = entry.2 := by
  revert member unique
  induction table with
  | nil => intro member _; cases member
  | cons first rest ih =>
      obtain ⟨stored, previous⟩ := first
      intro member unique
      have both : stored ∉ rest.map Prod.fst ∧ (rest.map Prod.fst).Nodup := by
        simpa only [List.map_cons, List.nodup_cons] using unique
      rcases List.mem_cons.mp member with equal | member
      · subst entry
        simp [count_values.lookup]
      · have different : stored ≠ entry.1 := by
          intro same
          apply both.1
          rw [same]
          exact List.mem_map.mpr ⟨entry, member, rfl⟩
        rw [count_values.lookup, if_neg different]
        exact ih member both.2

/-- Every actual frequency-table entry equals the corresponding occurrence count. -/
theorem count_entry (values : List Int) (entry : Int × Int) (member : entry ∈ count_values.counts values) :
    entry.2 = (values.count entry.1 : Int) := by
  have selected := lookup_of_mem (count_values.counts values) entry member (count_values.counts_keys_nodup values)
  exact selected.symm.trans (count_values.lookup_counts values entry.1)

/-- Every sorted source tuple decodes to one of the original dictionary entries. -/
theorem decoded_group_member (heap : Mutable.Heap) (address : Nat) (table : List (Int × Int))
    (found : heap[address]? = some (.dict (count_values.entries table))) (group : Mutable.Value)
    (member : group ∈ rank_groups.resultGroups heap table) :
    ∃ entry ∈ table, ProjectEulerSolutions.TupleSorting.readInts (rank_groups.collected heap table) group = [entry.2, entry.1] := by
  have original := (ProjectEulerSolutions.TupleSorting.sorted_perm (rank_groups.collected heap table) true
    (rank_groups.references (heap.length+1) table)).mem_iff.mp member
  have decoded : ProjectEulerSolutions.TupleSorting.readInts (rank_groups.collected heap table) group ∈
      (rank_groups.references (heap.length+1) table).map
        (ProjectEulerSolutions.TupleSorting.readInts (rank_groups.collected heap table)) :=
    List.mem_map.mpr ⟨group, original, rfl⟩
  rw [rank_groups.Represents.readInts _ _ _ (rank_groups.collected_represents heap address table found)] at decoded
  obtain ⟨entry, entryMember, equal⟩ := List.mem_map.mp decoded
  exact ⟨entry, entryMember, equal.symm⟩

theorem decoded_count_group (heap : Mutable.Heap) (address : Nat) (values : List Int)
    (found : heap[address]? = some (.dict (count_values.entries (count_values.counts values))))
    (group : Mutable.Value) (member : group ∈ rank_groups.resultGroups heap (count_values.counts values)) :
    ∃ rank : Int, ProjectEulerSolutions.TupleSorting.readInts (rank_groups.collected heap (count_values.counts values)) group =
      [(values.count rank : Int), rank] := by
  obtain ⟨entry, entryMember, decoded⟩ := decoded_group_member heap address (count_values.counts values) found group member
  refine ⟨entry.1, ?_⟩
  rw [decoded, count_entry values entry entryMember]

theorem group_count (heap : Mutable.Heap) (address : Nat) (values : List Int)
    (found : heap[address]? = some (.dict (count_values.entries (count_values.counts values))))
    (group : Mutable.Value) (member : group ∈ rank_groups.resultGroups heap (count_values.counts values))
    (frequency rank : Int)
    (decoded : ProjectEulerSolutions.TupleSorting.readInts (rank_groups.collected heap (count_values.counts values)) group =
      [frequency, rank]) :
    (values.count rank : Int) = frequency := by
  obtain ⟨actualRank, actual⟩ := decoded_count_group heap address values found group member
  have same : [(values.count actualRank : Int), actualRank] = [frequency, rank] := actual.symm.trans decoded
  have fields : (values.count actualRank : Int) = frequency ∧ actualRank = rank := by simpa using same
  rcases fields with ⟨counted, rfl⟩
  exact counted

/-- A four-of-a-kind tuple produced by the source has a genuine remaining kicker. -/
theorem four_group_kicker_nonempty (heap : Mutable.Heap) (address : Nat) (values : List Int)
    (found : heap[address]? = some (.dict (count_values.entries (count_values.counts values))))
    (five : values.length = 5) (group : Mutable.Value)
    (member : group ∈ rank_groups.resultGroups heap (count_values.counts values)) (four : Int)
    (decoded : ProjectEulerSolutions.TupleSorting.readInts (rank_groups.collected heap (count_values.counts values)) group = [4, four]) :
    values.filter (fun value => value != four) ≠ [] := by
  have counted := group_count heap address values found group member 4 four decoded
  exact four_kicker_nonempty values four five (by omega)

/-- Two distinct pair tuples produced by the source leave a fifth card. -/
theorem two_pair_groups_kicker_nonempty (heap : Mutable.Heap) (address : Nat) (values : List Int)
    (found : heap[address]? = some (.dict (count_values.entries (count_values.counts values))))
    (five : values.length = 5) (highGroup lowGroup : Mutable.Value)
    (highMember : highGroup ∈ rank_groups.resultGroups heap (count_values.counts values))
    (lowMember : lowGroup ∈ rank_groups.resultGroups heap (count_values.counts values)) (high low : Int)
    (different : high ≠ low)
    (highDecoded : ProjectEulerSolutions.TupleSorting.readInts (rank_groups.collected heap (count_values.counts values)) highGroup = [2, high])
    (lowDecoded : ProjectEulerSolutions.TupleSorting.readInts (rank_groups.collected heap (count_values.counts values)) lowGroup = [2, low]) :
    values.filter (fun value => value != high && value != low) ≠ [] := by
  have highCount := group_count heap address values found highGroup highMember 2 high highDecoded
  have lowCount := group_count heap address values found lowGroup lowMember 2 low lowDecoded
  exact two_pairs_kicker_nonempty values high low five different (by omega) (by omega)

end solvers.«54».Kickers

namespace solvers.«54».Kickers
open FormalSW.PyAst

def decodedRanks (heap : Mutable.Heap) (table : List (Int × Int)) : List Int :=
  ((rank_groups.resultGroups heap table).map
    (ProjectEulerSolutions.TupleSorting.readInts (rank_groups.collected heap table))).map (fun row => row.getD 1 0)

def returnedHeap (heap : Mutable.Heap) (table : List (Int × Int)) : Mutable.Heap :=
  rank_groups.collected heap table ++ [.list (rank_groups.resultGroups heap table)]

def returnedRanks (heap : Mutable.Heap) (table : List (Int × Int)) : List Int :=
  ((rank_groups.resultGroups heap table).map
    (ProjectEulerSolutions.TupleSorting.readInts (returnedHeap heap table))).map (fun row => row.getD 1 0)

/-- Actual sorted source tuples retain exactly the original dictionary ranks. -/
theorem decodedRanks_perm (heap : Mutable.Heap) (address : Nat) (table : List (Int × Int))
    (found : heap[address]? = some (.dict (count_values.entries table))) :
    (decodedRanks heap table).Perm (table.map Prod.fst) := by
  have permutation := ((ProjectEulerSolutions.TupleSorting.sorted_perm (rank_groups.collected heap table) true
    (rank_groups.references (heap.length+1) table)).map
      (ProjectEulerSolutions.TupleSorting.readInts (rank_groups.collected heap table))).map (fun row => row.getD 1 0)
  have rows := rank_groups.Represents.readInts _ _ _ (rank_groups.collected_represents heap address table found)
  rw [rows] at permutation
  simpa only [decodedRanks, List.map_map, Function.comp_def] using permutation

theorem returnedRanks_eq (heap : Mutable.Heap) (address : Nat) (table : List (Int × Int))
    (found : heap[address]? = some (.dict (count_values.entries table))) :
    returnedRanks heap table = decodedRanks heap table := by
  exact congrArg (List.map (fun row : List Int => row.getD 1 0))
    (rank_groups.resultGroups_readInts_append heap address table found [.list (rank_groups.resultGroups heap table)])

theorem returnedRanks_perm (heap : Mutable.Heap) (address : Nat) (table : List (Int × Int))
    (found : heap[address]? = some (.dict (count_values.entries table))) :
    (returnedRanks heap table).Perm (table.map Prod.fst) := by
  rw [returnedRanks_eq heap address table found]
  exact decodedRanks_perm heap address table found

/-- In the final source heap, frequency groups still have pairwise distinct ranks. -/
theorem returnedRanks_nodup (heap : Mutable.Heap) (address : Nat) (values : List Int)
    (found : heap[address]? = some (.dict (count_values.entries (count_values.counts values)))) :
    (returnedRanks heap (count_values.counts values)).Nodup :=
  (returnedRanks_perm heap address (count_values.counts values) found).nodup_iff.mpr
    (count_values.counts_keys_nodup values)

theorem returned_row_eq (heap : Mutable.Heap) (address : Nat) (table : List (Int × Int))
    (found : heap[address]? = some (.dict (count_values.entries table))) (group : Mutable.Value)
    (member : group ∈ rank_groups.resultGroups heap table) :
    ProjectEulerSolutions.TupleSorting.readInts (returnedHeap heap table) group =
      ProjectEulerSolutions.TupleSorting.readInts (rank_groups.collected heap table) group := by
  obtain ⟨index, atIndex⟩ := List.mem_iff_getElem?.mp member
  have equal := congrArg (fun rows : List (List Int) => rows[index]?)
    (rank_groups.resultGroups_readInts_append heap address table found [.list (rank_groups.resultGroups heap table)])
  simpa only [returnedHeap, List.getElem?_map, atIndex, Option.map_some, Option.some.injEq] using equal

/-- Two different successful group subscriptions cannot name the same rank. -/
theorem group_rank_ne_of_indices (heap : Mutable.Heap) (address : Nat) (values : List Int)
    (found : heap[address]? = some (.dict (count_values.entries (count_values.counts values))))
    (i j : Nat) (left right : Mutable.Value)
    (atLeft : (rank_groups.resultGroups heap (count_values.counts values))[i]? = some left)
    (atRight : (rank_groups.resultGroups heap (count_values.counts values))[j]? = some right)
    (different : i ≠ j) :
    (ProjectEulerSolutions.TupleSorting.readInts (returnedHeap heap (count_values.counts values)) left).getD 1 0 ≠
      (ProjectEulerSolutions.TupleSorting.readInts (returnedHeap heap (count_values.counts values)) right).getD 1 0 := by
  have leftRead : (returnedRanks heap (count_values.counts values))[i]? = some
      ((ProjectEulerSolutions.TupleSorting.readInts (returnedHeap heap (count_values.counts values)) left).getD 1 0) := by
    simp only [returnedRanks, List.getElem?_map, atLeft, Option.map_some]
  have rightRead : (returnedRanks heap (count_values.counts values))[j]? = some
      ((ProjectEulerSolutions.TupleSorting.readInts (returnedHeap heap (count_values.counts values)) right).getD 1 0) := by
    simp only [returnedRanks, List.getElem?_map, atRight, Option.map_some]
  intro same
  apply different
  apply (List.getElem?_inj (List.getElem?_eq_some_iff.mp leftRead).1 (returnedRanks_nodup heap address values found)).mp
  rw [leftRead, rightRead, same]

/-- The actual two-pair source positions imply both distinct ranks and a nonempty kicker filter. -/
theorem two_pair_positions_kicker_nonempty (heap : Mutable.Heap) (address : Nat) (values : List Int)
    (found : heap[address]? = some (.dict (count_values.entries (count_values.counts values))))
    (five : values.length = 5) (i j : Nat) (highGroup lowGroup : Mutable.Value)
    (atHigh : (rank_groups.resultGroups heap (count_values.counts values))[i]? = some highGroup)
    (atLow : (rank_groups.resultGroups heap (count_values.counts values))[j]? = some lowGroup)
    (different : i ≠ j) (high low : Int)
    (highDecoded : ProjectEulerSolutions.TupleSorting.readInts (returnedHeap heap (count_values.counts values)) highGroup = [2, high])
    (lowDecoded : ProjectEulerSolutions.TupleSorting.readInts (returnedHeap heap (count_values.counts values)) lowGroup = [2, low]) :
    values.filter (fun value => value != high && value != low) ≠ [] := by
  have highMember := List.mem_iff_getElem?.mpr ⟨i, atHigh⟩
  have lowMember := List.mem_iff_getElem?.mpr ⟨j, atLow⟩
  have separated := group_rank_ne_of_indices heap address values found i j highGroup lowGroup atHigh atLow different
  have distinctRanks : high ≠ low := by
    simpa only [highDecoded, lowDecoded, List.getD_cons_succ, List.getD_cons_zero] using separated
  have highBefore : ProjectEulerSolutions.TupleSorting.readInts (rank_groups.collected heap (count_values.counts values)) highGroup =
      [2, high] := (returned_row_eq heap address (count_values.counts values) found highGroup highMember).symm.trans highDecoded
  have lowBefore : ProjectEulerSolutions.TupleSorting.readInts (rank_groups.collected heap (count_values.counts values)) lowGroup =
      [2, low] := (returned_row_eq heap address (count_values.counts values) found lowGroup lowMember).symm.trans lowDecoded
  exact two_pair_groups_kicker_nonempty heap address values found five highGroup lowGroup highMember lowMember high low
    distinctRanks highBefore lowBefore

end solvers.«54».Kickers

namespace solvers.«54».numeric_score.straight_detection
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
set_option Elab.async false

private abbrev program := Solvers.P54Ast.module
private abbrev number (n : Int) : Mutable.Value := .scalar (.int n)
private abbrev numbers (values : List Int) : List Mutable.Value := values.map number
private abbrev flag (b : Bool) : Mutable.Value := .scalar (.bool b)
private def code : List Stmt := executableBody ((findFunctionIn program.body "_score_values").getD ([], [])).2

def wheel : List Int := [2, 3, 4, 5, 14]
def spread (values : List Int) : Int := values.getLastD 0 - values.headD 0
def detected (values : List Int) : Bool := decide (values.length = 5 ∧ (spread values = 4 ∨ values = wheel))
def high (values : List Int) : Int := if detected values then if spread values = 4 then values.getLastD 0 else 5 else 0
def updated (env : HeapCalls.Env) (values : List Int) : HeapCalls.Env :=
  if detected values then setLocal (setLocal env "is_straight" (flag true)) "straight_high" (number (high values)) else env
def extra (values : List Int) : Heap :=
  if values.length = 5 ∧ spread values ≠ 4 then [.list (numbers wheel)] else []

private abbrev lengthTest : Expr := .compare (.builtinCall .len [.name "uniq"]) [(.eq, .constant (.int 5))]
private abbrev firstExpression : Expr := .subscript (.name "uniq") (.constant (.int 0))
private abbrev lastExpression : Expr := .subscript (.name "uniq") (.unaryOp .uSub (.constant (.int 1)))
private abbrev spreadTest : Expr := .compare (.binOp lastExpression .sub firstExpression) [(.eq, .constant (.int 4))]
private abbrev wheelExpression : Expr := .list
  [.constant (.int 2), .constant (.int 3), .constant (.int 4), .constant (.int 5), .constant (.int 14)]
private abbrev wheelTest : Expr := .compare (.name "uniq") [(.eq, wheelExpression)]
private abbrev assignments (top : Expr) : List Stmt :=
  [.assign [.name "is_straight"] (.constant (.bool true)), .assign [.name "straight_high"] top]
private abbrev statement : Stmt := .ifThen lengthTest
  [.ifThen spreadTest (assignments lastExpression) [.ifThen wheelTest (assignments (.constant (.int 5))) []]] []

private theorem found_append {heap : Heap} {address : Nat} {object : Mutable.Object}
    (found : heap[address]? = some object) (objects : Heap) : (heap ++ objects)[address]? = some object := by
  rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1]
  exact found

private theorem length_test (heap : Heap) (env : HeapCalls.Env) (address : Nat) (values : List Int)
    (globals : Mutable.Globals) (bound : env "uniq" = some (.ref address))
    (found : heap[address]? = some (.list (numbers values))) :
    Evaluates program env lengthTest heap (flag (decide (values.length = 5))) heap globals := by
  apply Evaluates.compare (av := number values.length) (bv := number 5) (middle := heap) (afterRight := heap)
  · apply Evaluates.builtin (.cons (Evaluates.lookup bound) .nil)
    simp [primitive, Mutable.length, found, numbers]
  · exact Evaluates.constant
  · simp [HeapCalls.compare, scalarCompare, evalCmpOp, flag]
    omega

private theorem first_five (heap : Heap) (env : HeapCalls.Env) (address : Nat) (a b c d e : Int)
    (globals : Mutable.Globals) (bound : env "uniq" = some (.ref address))
    (found : heap[address]? = some (.list (numbers [a, b, c, d, e]))) :
    Evaluates program env firstExpression heap (number a) heap globals := by
  apply Evaluates.subscript (by intros; simp) (Evaluates.lookup bound) Evaluates.constant
  simp [Mutable.getItem, Mutable.index, found, numbers, number, Except.map]

private theorem last_five (heap : Heap) (env : HeapCalls.Env) (address : Nat) (a b c d e : Int)
    (globals : Mutable.Globals) (bound : env "uniq" = some (.ref address))
    (found : heap[address]? = some (.list (numbers [a, b, c, d, e]))) :
    Evaluates program env lastExpression heap (number e) heap globals := by
  apply Evaluates.subscript (by intros; simp) (Evaluates.lookup bound) (Evaluates.unary Evaluates.constant rfl)
  simp [Mutable.getItem, Mutable.index, found, numbers, number, Except.map]

private theorem spread_test (heap : Heap) (env : HeapCalls.Env) (address : Nat) (a b c d e : Int)
    (globals : Mutable.Globals) (bound : env "uniq" = some (.ref address))
    (found : heap[address]? = some (.list (numbers [a, b, c, d, e]))) :
    Evaluates program env spreadTest heap (flag (decide (e-a = 4))) heap globals := by
  apply Evaluates.compare (av := number (e-a)) (bv := number 4) (middle := heap) (afterRight := heap)
  · exact Evaluates.binary (last_five heap env address a b c d e globals bound found)
      (first_five heap env address a b c d e globals bound found) rfl
  · exact Evaluates.constant
  · simp [HeapCalls.compare, scalarCompare, evalCmpOp, flag]

private theorem wheel_test (heap : Heap) (env : HeapCalls.Env) (address : Nat) (values : List Int)
    (globals : Mutable.Globals) (bound : env "uniq" = some (.ref address))
    (found : heap[address]? = some (.list (numbers values))) :
    Evaluates program env wheelTest heap (flag (decide (values = wheel)))
      (heap ++ [.list (numbers wheel)]) globals := by
  let middle := heap ++ [Mutable.Object.list (numbers wheel)]
  have inside := (List.getElem?_eq_some_iff.mp found).1
  have different : address ≠ heap.length := by omega
  have original : middle[address]? = some (.list (numbers values)) := found_append found _
  have literal : middle[heap.length]? = some (.list (numbers wheel)) := by simp [middle]
  apply Evaluates.compare (av := .ref address) (bv := .ref heap.length) (middle := heap) (afterRight := middle)
  · exact Evaluates.lookup bound
  · exact Evaluates.list (.cons Evaluates.constant (.cons Evaluates.constant
      (.cons Evaluates.constant (.cons Evaluates.constant (.cons Evaluates.constant .nil)))))
  · simp [HeapCalls.compare, Mutable.array, original, literal, different, equalLists_ints, numbers, flag]
    exact ⟨rfl, rfl⟩

private theorem five_parts (values : List Int) (five : values.length = 5) :
    ∃ a b c d e, values = [a, b, c, d, e] := by
  cases values with
  | nil => simp at five
  | cons a values => cases values with
    | nil => simp at five
    | cons b values => cases values with
      | nil => simp at five
      | cons c values => cases values with
        | nil => simp at five
        | cons d values => cases values with
          | nil => simp at five
          | cons e values =>
              have empty : values = [] := List.length_eq_zero_iff.mp (by simpa using five)
              exact ⟨a, b, c, d, e, by simp [empty]⟩

private theorem five_case (heap : Heap) (env : HeapCalls.Env) (address : Nat) (a b c d e : Int)
    (globals : Mutable.Globals) (bound : env "uniq" = some (.ref address))
    (found : heap[address]? = some (.list (numbers [a, b, c, d, e]))) :
    BlockContinues program env [statement] heap (updated env [a, b, c, d, e])
      (heap ++ extra [a, b, c, d, e]) globals := by
  by_cases range : e-a = 4
  · simp [updated, detected, high, extra, spread, range]
    apply BlockContinues.branch (flag := true) (value := flag true) (middle := heap)
    · simpa using length_test heap env address [a, b, c, d, e] globals bound found
    · rfl
    · apply BlockContinues.branch (flag := true) (value := flag true) (middle := heap)
      · simpa only [range, decide_true] using spread_test heap env address a b c d e globals bound found
      · rfl
      · exact (BlockContinues.assign (name := "is_straight") (Evaluates.constant (value := .bool true))).append
          (BlockContinues.assign (name := "straight_high")
            (last_five heap (setLocal env "is_straight" (flag true)) address a b c d e globals
              (by simpa [setLocal] using bound) found))
  · by_cases special : [a, b, c, d, e] = wheel
    · simp [updated, detected, high, extra, spread, special]
      apply BlockContinues.branch (flag := true) (value := flag true) (middle := heap)
      · simpa using length_test heap env address [a, b, c, d, e] globals bound found
      · rfl
      · apply BlockContinues.branch (flag := false) (value := flag false) (middle := heap)
        · simpa only [range, decide_false] using spread_test heap env address a b c d e globals bound found
        · rfl
        · apply BlockContinues.branch (flag := true) (value := flag true) (middle := heap ++ [.list (numbers wheel)])
          · simpa only [special, decide_true] using wheel_test heap env address [a, b, c, d, e] globals bound found
          · rfl
          · exact (BlockContinues.assign (name := "is_straight") (Evaluates.constant (value := .bool true))).append
              (BlockContinues.assign (name := "straight_high") (Evaluates.constant (value := .int 5)))
    · simp [updated, detected, extra, spread, range, special]
      apply BlockContinues.branch (flag := true) (value := flag true) (middle := heap)
      · simpa using length_test heap env address [a, b, c, d, e] globals bound found
      · rfl
      · apply BlockContinues.branch (flag := false) (value := flag false) (middle := heap)
        · simpa only [range, decide_false] using spread_test heap env address a b c d e globals bound found
        · rfl
        · apply BlockContinues.branch (flag := false) (value := flag false) (middle := heap ++ [.list (numbers wheel)])
          · simpa only [special, decide_false] using wheel_test heap env address [a, b, c, d, e] globals bound found
          · rfl
          · exact BlockContinues.nil

/-- The source handles every unique-list length safely, allocating the wheel literal exactly when needed. -/
theorem result (heap : Heap) (env : HeapCalls.Env) (address : Nat) (values : List Int)
    (globals : Mutable.Globals) (bound : env "uniq" = some (.ref address))
    (found : heap[address]? = some (.list (numbers values))) :
    BlockContinues program env ((code.drop 5).take 1) heap (updated env values) (heap ++ extra values) globals := by
  change BlockContinues program env [statement] heap (updated env values) (heap ++ extra values) globals
  by_cases five : values.length = 5
  · obtain ⟨a, b, c, d, e, rfl⟩ := five_parts values five
    exact five_case heap env address a b c d e globals bound found
  · simp only [updated, detected, extra, five, false_and, decide_false, Bool.false_eq_true, ↓reduceIte, List.append_nil]
    apply BlockContinues.branch (flag := false) (value := flag false) (middle := heap)
    · simpa only [five, decide_false] using length_test heap env address values globals bound found
    · rfl
    · exact BlockContinues.nil

theorem updated_straight (env : HeapCalls.Env) (values : List Int)
    (initial : env "is_straight" = some (flag false)) :
    updated env values "is_straight" = some (flag (detected values)) := by
  cases test : detected values <;> simp [updated, test, setLocal, initial]

theorem updated_high (env : HeapCalls.Env) (values : List Int)
    (initial : env "straight_high" = some (number 0)) :
    updated env values "straight_high" = some (number (high values)) := by
  cases test : detected values <;> simp [updated, high, test, setLocal, initial]

theorem updated_other (env : HeapCalls.Env) (values : List Int) (name : String)
    (notStraight : name ≠ "is_straight") (notHigh : name ≠ "straight_high") :
    updated env values name = env name := by
  cases test : detected values <;> simp [updated, test, setLocal, notStraight, notHigh]

/-- Composition with the checked normalization phase executes the first six source statements. -/
theorem normalization_straight (heap : Heap) (env : HeapCalls.Env) (valuesAddress suitsAddress : Nat)
    (values : List Int) (suits : List String) (globals : Mutable.Globals)
    (nonempty : values ≠ [])
    (valuesFound : heap[valuesAddress]? = some (.list (numbers values)))
    (suitsFound : heap[suitsAddress]? = some (.list (suits.map fun value => Mutable.Value.scalar (.string value))))
    (getValues : env "values" = some (.ref valuesAddress)) (getSuits : env "suits" = some (.ref suitsAddress)) :
    BlockContinues program env (code.take 6) heap
      (updated (normalizationEnv env heap suits) (orderedUnique values))
      (normalizationHeap heap valuesAddress values suits ++ extra (orderedUnique values)) globals := by
  have normalized := normalization heap env valuesAddress suitsAddress values suits globals nonempty valuesFound suitsFound getValues getSuits
  have found : (normalizationHeap heap valuesAddress values suits)[heap.length+2]? =
      some (.list (numbers (orderedUnique values))) := by
    simp [normalizationHeap]
  have branch := result (normalizationHeap heap valuesAddress values suits) (normalizationEnv env heap suits)
    (heap.length+2) (orderedUnique values) globals (by simp [normalizationEnv, setLocal]) found
  simpa [code] using normalized.append branch

private abbrev sourceArgs := ((findFunctionIn program.body "_score_values").getD ([], [])).1
private abbrev sourceBody := ((findFunctionIn program.body "_score_values").getD ([], [])).2
private abbrev initial (globals : Mutable.Globals) (valuesAddress suitsAddress : Nat) : HeapCalls.Env :=
  functionEnv sourceArgs sourceBody [.ref valuesAddress, .ref suitsAddress] globals

/-- The complete numeric helper returns its real rank tuple on the straight-flush path. -/
theorem straight_flush_result (heap : Heap) (valuesAddress suitsAddress : Nat)
    (values : List Int) (suits : List String) (globals : Mutable.Globals)
    (nonempty : values ≠ [])
    (valuesFound : heap[valuesAddress]? = some (.list (numbers values)))
    (suitsFound : heap[suitsAddress]? = some (.list (suits.map fun value => Mutable.Value.scalar (.string value))))
    (straight : detected (orderedUnique values) = true) (flushing : isFlush suits = true) :
    let readyHeap := normalizationHeap heap valuesAddress values suits ++ extra (orderedUnique values)
    ReturnsValue program "_score_values" [.ref valuesAddress, .ref suitsAddress] heap (.ref readyHeap.length)
      (readyHeap ++ [.tuple [number 8, number (high (orderedUnique values))]]) globals := by
  let start := initial globals valuesAddress suitsAddress
  let ready := updated (normalizationEnv start heap suits) (orderedUnique values)
  let readyHeap := normalizationHeap heap valuesAddress values suits ++ extra (orderedUnique values)
  have getValues : start "values" = some (.ref valuesAddress) := by cases globals <;> rfl
  have getSuits : start "suits" = some (.ref suitsAddress) := by cases globals <;> rfl
  have normalized := normalization_straight heap start valuesAddress suitsAddress values suits globals
    nonempty valuesFound suitsFound getValues getSuits
  have getStraight : ready "is_straight" = some (flag true) := by
    simpa only [straight] using updated_straight (normalizationEnv start heap suits) (orderedUnique values)
      (by simp [normalizationEnv, setLocal])
  have getHigh : ready "straight_high" = some (number (high (orderedUnique values))) :=
    updated_high (normalizationEnv start heap suits) (orderedUnique values) (by simp [normalizationEnv, setLocal])
  have getFlush : ready "is_flush" = some (flag true) := by
    change updated (normalizationEnv start heap suits) (orderedUnique values) "is_flush" = _
    rw [updated_other _ _ "is_flush" (by decide) (by decide)]
    simp [normalizationEnv, setLocal, flushing]
  have returned : BlockReturns program ready (code.drop 6) readyHeap (.ref readyHeap.length)
      (readyHeap ++ [.tuple [number 8, number (high (orderedUnique values))]]) globals := by
    apply BlockReturns.branch (flag := true) (value := flag true) (middle := readyHeap)
    · exact Evaluates.boolean_next (flag := true) (Evaluates.lookup getStraight) rfl rfl
        (Evaluates.boolean_last (flag := true) (Evaluates.lookup getFlush) rfl)
    · rfl
    · apply BlockReturns.return_value
      exact Evaluates.tuple (.cons Evaluates.constant (.cons (Evaluates.lookup getHigh) .nil)) (by simp)
  apply BlockReturns.function rfl rfl
  simpa only [List.take_append_drop] using normalized.returns returned

def countedHeap (heap : Heap) (values : List Int) : Heap :=
  heap ++ [.dict (count_values.entries (count_values.counts values))]

def groupedHeap (heap : Heap) (values : List Int) : Heap :=
  rank_groups.collected (countedHeap heap values) (count_values.counts values) ++
    [.list (rank_groups.resultGroups (countedHeap heap values) (count_values.counts values))]

def groupedEnv (env : HeapCalls.Env) (heap : Heap) (values : List Int) : HeapCalls.Env :=
  setLocal (setLocal env "cnt" (.ref heap.length)) "groups"
    (.ref (rank_groups.collected (countedHeap heap values) (count_values.counts values)).length)

private theorem straight_flush_test (heap : Heap) (env : HeapCalls.Env) (straight flush : Bool)
    (globals : Mutable.Globals)
    (getStraight : env "is_straight" = some (flag straight)) (getFlush : env "is_flush" = some (flag flush)) :
    Evaluates program env (.boolOp .and [.name "is_straight", .name "is_flush"])
      heap (flag (straight && flush)) heap globals := by
  cases straight with
  | false => exact Evaluates.boolean_stop (flag := false) (Evaluates.lookup getStraight) rfl rfl
  | true =>
      exact Evaluates.boolean_next (flag := true) (Evaluates.lookup getStraight) rfl rfl
        (Evaluates.boolean_last (flag := flush) (Evaluates.lookup getFlush) rfl)

/-- After the non-straight-flush branch, the source counts ranks and returns the real sorted tuple groups. -/
theorem groups_phase (heap : Heap) (env : HeapCalls.Env) (valuesAddress : Nat) (values : List Int)
    (straight flush : Bool) (globals : Mutable.Globals)
    (found : heap[valuesAddress]? = some (.list (numbers values)))
    (getValues : env "values" = some (.ref valuesAddress))
    (getStraight : env "is_straight" = some (flag straight)) (getFlush : env "is_flush" = some (flag flush))
    (notBoth : (straight && flush) = false) :
    BlockContinues program env ((code.drop 6).take 3) heap (groupedEnv env heap values) (groupedHeap heap values) globals := by
  let counted := countedHeap heap values
  let counterEnv := setLocal env "cnt" (.ref heap.length)
  have countsFound : counted[heap.length]? = some (.dict (count_values.entries (count_values.counts values))) := by
    simp [counted, countedHeap]
  have skipped : BlockContinues program env [(code[6]?).getD (.expr (.constant .none))] heap env heap globals := by
    apply BlockContinues.branch (flag := false) (value := flag false) (middle := heap)
    · simpa only [notBoth] using straight_flush_test heap env straight flush globals getStraight getFlush
    · rfl
    · exact BlockContinues.nil
  have counts := BlockContinues.assign (name := "cnt")
    (Evaluates.call (.cons (Evaluates.lookup getValues) .nil)
      (count_values.result heap valuesAddress values globals found))
  have groups := BlockContinues.assign (name := "groups")
    (Evaluates.call (.cons (Evaluates.lookup (env := counterEnv) (name := "cnt") rfl) .nil)
      (rank_groups.result counted heap.length (count_values.counts values) globals countsFound))
  simpa [code, groupedEnv, groupedHeap, counted, countedHeap, counterEnv] using skipped.append (counts.append groups)

/-- The profile certificate applies to the very heap returned by the operational group phase. -/
theorem grouped_profile (heap : Heap) (values : List Int) (five : values.length = 5) :
    ((rank_groups.resultGroups (countedHeap heap values) (count_values.counts values)).map
      (ProjectEulerSolutions.TupleSorting.readInts (groupedHeap heap values))).map (fun row => row.headD 0) ∈
        ProjectEulerSolutions.PokerCountPatterns.profiles := by
  exact rank_groups.resultGroups_profile_returned (countedHeap heap values) heap.length values
    (by simp [countedHeap]) five

theorem grouped_other (env : HeapCalls.Env) (heap : Heap) (values : List Int) (name : String)
    (notCounts : name ≠ "cnt") (notGroups : name ≠ "groups") :
    groupedEnv env heap values name = env name := by
  simp [groupedEnv, setLocal, notCounts, notGroups]

private theorem represented_members (heap : Heap) (groups : List Mutable.Value) (table : List (Int × Int))
    (rows : rank_groups.Represents heap groups table) :
    ∀ group ∈ groups, ∃ pair ∈ table, ∃ address : Nat,
      group = .ref address ∧ heap[address]? = some (rank_groups.groupTuple pair) := by
  induction rows with
  | nil => intro group member; cases member
  | @cons group pair groups table first rest ih =>
      intro value member
      rcases List.mem_cons.mp member with same | member
      · subst value
        obtain ⟨address, reference, found⟩ := first
        exact ⟨pair, List.mem_cons_self, address, reference, found⟩
      · obtain ⟨other, otherMember, address, reference, found⟩ := ih value member
        exact ⟨other, List.mem_cons_of_mem pair otherMember, address, reference, found⟩

/-- Each returned group is an actual two-field tuple from an entry of the exact source count dictionary. -/
theorem grouped_member (heap : Heap) (values : List Int) (group : Mutable.Value)
    (member : group ∈ rank_groups.resultGroups (countedHeap heap values) (count_values.counts values)) :
    ∃ pair ∈ count_values.counts values, ∃ address : Nat,
      group = .ref address ∧ (groupedHeap heap values)[address]? = some (.tuple [number pair.2, number pair.1]) := by
  let counted := countedHeap heap values
  let table := count_values.counts values
  have countsFound : counted[heap.length]? = some (.dict (count_values.entries table)) := by simp [counted, countedHeap, table]
  have rows := rank_groups.collected_represents counted heap.length table countsFound
  have original : group ∈ rank_groups.references (counted.length+1) table :=
    (ProjectEulerSolutions.TupleSorting.sorted_perm (rank_groups.collected counted table) true
      (rank_groups.references (counted.length+1) table)).mem_iff.mp member
  obtain ⟨pair, pairMember, address, reference, found⟩ := represented_members
    (rank_groups.collected counted table) (rank_groups.references (counted.length+1) table) table rows group original
  exact ⟨pair, pairMember, address, reference, found_append found _⟩

theorem grouped_found (heap : Heap) (values : List Int) :
    (groupedHeap heap values)[(rank_groups.collected (countedHeap heap values) (count_values.counts values)).length]? =
      some (.list (rank_groups.resultGroups (countedHeap heap values) (count_values.counts values))) := by
  simp [groupedHeap]

def groupExpression (index field : Nat) : Expr :=
  .subscript (.subscript (.name "groups") (.constant (.int index))) (.constant (.int field))

private theorem group_item (heap : Heap) (env : HeapCalls.Env) (listAddress index tupleAddress : Nat)
    (groups : List Mutable.Value) (globals : Mutable.Globals)
    (bound : env "groups" = some (.ref listAddress)) (found : heap[listAddress]? = some (.list groups))
    (element : groups[index]? = some (.ref tupleAddress)) :
    Evaluates program env (.subscript (.name "groups") (.constant (.int index))) heap (.ref tupleAddress) heap globals := by
  apply Evaluates.subscript (by intros; simp) (Evaluates.lookup bound) Evaluates.constant
  obtain ⟨inside, item⟩ := List.getElem?_eq_some_iff.mp element
  simpa only [item] using Mutable.getItem_list heap listAddress index groups found inside

/-- Reading a count follows the actual groups list reference and its immutable source pair. -/
theorem group_count (heap : Heap) (env : HeapCalls.Env) (listAddress index tupleAddress : Nat)
    (groups : List Mutable.Value) (count rank : Int) (globals : Mutable.Globals)
    (bound : env "groups" = some (.ref listAddress)) (found : heap[listAddress]? = some (.list groups))
    (element : groups[index]? = some (.ref tupleAddress))
    (pair : heap[tupleAddress]? = some (.tuple [number count, number rank])) :
    Evaluates program env (groupExpression index 0) heap (number count) heap globals := by
  apply Evaluates.subscript (by intros; simp)
    (group_item heap env listAddress index tupleAddress groups globals bound found element) Evaluates.constant
  simp [Mutable.getItem, Mutable.index, pair, number, Except.map]

theorem group_rank (heap : Heap) (env : HeapCalls.Env) (listAddress index tupleAddress : Nat)
    (groups : List Mutable.Value) (count rank : Int) (globals : Mutable.Globals)
    (bound : env "groups" = some (.ref listAddress)) (found : heap[listAddress]? = some (.list groups))
    (element : groups[index]? = some (.ref tupleAddress))
    (pair : heap[tupleAddress]? = some (.tuple [number count, number rank])) :
    Evaluates program env (groupExpression index 1) heap (number rank) heap globals := by
  apply Evaluates.subscript (by intros; simp)
    (group_item heap env listAddress index tupleAddress groups globals bound found element) Evaluates.constant
  simp [Mutable.getItem, Mutable.index, pair, number, Except.map]

end solvers.«54».numeric_score.straight_detection

namespace solvers.«54».numeric_score.straight_detection

open FormalSW.PyAst

set_option Elab.async false

private theorem five_entries (values : List Int) (five : values.length = 5) :
    ∃ a b c d e : Int, values = [a, b, c, d, e] := by
  cases values with
  | nil => simp at five
  | cons a rest =>
    cases rest with
    | nil => simp at five
    | cons b rest =>
      cases rest with
      | nil => simp at five
      | cons c rest =>
        cases rest with
        | nil => simp at five
        | cons d rest =>
          cases rest with
          | nil => simp at five
          | cons e rest =>
            have empty : rest.length = 0 := by
              simp only [List.length_cons] at five
              omega
            have restEmpty : rest = [] := List.length_eq_zero_iff.mp empty
            exact ⟨a, b, c, d, e, by rw [restEmpty]⟩

/-- The source detector and the indexed reference detector agree on five strictly ordered ranks. -/
theorem detected_eq_reference_of_five (values : List Int) (five : values.length = 5)
    (ordered : values.Pairwise (fun left right : Int => left < right)) :
    detected values = ProjectEulerSolutions.PokerStraights.referenceStraight values := by
  obtain ⟨a, b, c, d, e, same⟩ := five_entries values five
  subst values
  have first := List.pairwise_cons.mp ordered
  have second := List.pairwise_cons.mp first.2
  have third := List.pairwise_cons.mp second.2
  have fourth := List.pairwise_cons.mp third.2
  have ab : a < b := first.1 b (by simp)
  have bc : b < c := second.1 c (by simp)
  have cd : c < d := third.1 d (by simp)
  have de : d < e := fourth.1 e (by simp)
  rw [ProjectEulerSolutions.PokerStraights.five_straight a b c d e ab bc cd de]
  simp [detected, spread, wheel, ProjectEulerSolutions.PokerStraights.wheel]

/-- Deduplication excludes repeated-rank straights, and the remaining five-rank tests coincide. -/
theorem detected_orderedUnique_eq_reference (values : List Int) (five : values.length = 5) :
    detected (orderedUnique values) =
      ProjectEulerSolutions.PokerStraights.referenceStraight (ScalarSort.ints values) := by
  change detected (ProjectEulerSolutions.PokerUnique.orderedUnique values) = _
  rw [ProjectEulerSolutions.PokerUnique.five_referenceStraight values five]
  by_cases complete : (ProjectEulerSolutions.PokerUnique.orderedUnique values).length = 5
  · rw [if_pos complete]
    exact detected_eq_reference_of_five _ complete
      (ProjectEulerSolutions.PokerUnique.orderedUnique_strict values)
  · rw [if_neg complete]
    simp [detected, complete]

/-- Once the source detector succeeds, its width branch and wheel branch return the reference high. -/
theorem high_eq_reference_of_detected (values : List Int) (formed : detected values = true) :
    high values = ProjectEulerSolutions.PokerStraights.straightHigh values := by
  have conditions : values.length = 5 ∧ (spread values = 4 ∨ values = wheel) :=
    of_decide_eq_true formed
  obtain ⟨a, b, c, d, e, same⟩ := five_entries values conditions.1
  subst values
  by_cases width : spread [a, b, c, d, e] = 4
  · have endpoint : e - a = 4 := width
    calc
      high [a, b, c, d, e] = e := by simp [high, formed, width]
      _ = ProjectEulerSolutions.PokerStraights.straightHigh [a, b, c, d, e] :=
        (ProjectEulerSolutions.PokerStraights.width_high a b c d e endpoint).symm
  · have special : [a, b, c, d, e] = wheel := conditions.2.resolve_left width
    have pureSpecial : [a, b, c, d, e] = ProjectEulerSolutions.PokerStraights.wheel := special
    calc
      high [a, b, c, d, e] = 5 := by simp [high, formed, width]
      _ = ProjectEulerSolutions.PokerStraights.straightHigh [a, b, c, d, e] := by
        rw [pureSpecial, ProjectEulerSolutions.PokerStraights.wheel_high]

/-- The numeric scorer and reference scorer use the same straight high for every five-card input. -/
theorem high_orderedUnique_eq_reference (values : List Int) (five : values.length = 5)
    (formed : detected (orderedUnique values) = true) :
    high (orderedUnique values) =
      ProjectEulerSolutions.PokerStraights.straightHigh (ScalarSort.ints values) := by
  change detected (ProjectEulerSolutions.PokerUnique.orderedUnique values) = true at formed
  change high (ProjectEulerSolutions.PokerUnique.orderedUnique values) = _
  have uniqueFive : (ProjectEulerSolutions.PokerUnique.orderedUnique values).length = 5 :=
    (of_decide_eq_true formed).1
  rw [ProjectEulerSolutions.PokerUnique.sorted_eq_orderedUnique_five values five uniqueFive]
  exact high_eq_reference_of_detected _ formed

end solvers.«54».numeric_score.straight_detection
