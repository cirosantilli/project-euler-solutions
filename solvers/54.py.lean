import data.«project-euler-statements».data.py.project_euler_statements.«p54.py»
import «.formalsw».solvers.«54.py.ast»
import lang.py.FormalSW.DictionaryCalls
import lang.py.FormalSW.DictionaryItemsCalls
import lang.py.FormalSW.ReverseSortingCalls
import lang.py.FormalSW.SetCalls
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

namespace solvers.«54».numeric_score.simple_categories
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
set_option Elab.async false

private abbrev program := Solvers.P54Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
private abbrev numbers (values : List Int) : List Mutable.Value := values.map number
private abbrev flag (value : Bool) : Mutable.Value := .scalar (.bool value)
private def code : List Stmt := executableBody ((findFunctionIn program.body "_score_values").getD ([], [])).2

private theorem equal_integer (heap : Heap) (env : HeapCalls.Env) (expression : Expr) (value wanted : Int)
    (globals : Mutable.Globals) (evaluated : Evaluates program env expression heap (number value) heap globals) :
    Evaluates program env (.compare expression [(.eq, .constant (.int wanted))])
      heap (flag (decide (value = wanted))) heap globals := by
  apply Evaluates.compare (av := number value) (bv := number wanted) (middle := heap) (afterRight := heap)
    evaluated Evaluates.constant
  simp [HeapCalls.compare, scalarCompare, evalCmpOp, flag]

/-- A nonempty values buffer becomes one fresh tuple without changing any existing object. -/
private theorem values_tuple (heap : Heap) (env : HeapCalls.Env) (address : Nat) (values : List Int)
    (globals : Mutable.Globals) (nonempty : values ≠ [])
    (bound : env "values" = some (.ref address)) (found : heap[address]? = some (.list (numbers values))) :
    Evaluates program env (.builtinCall .tuple [.name "values"]) heap (.ref heap.length)
      (heap ++ [.tuple (numbers values)]) globals := by
  apply Evaluates.builtin (.cons (Evaluates.lookup bound) .nil)
  have converted := primitive_tuple_list heap address (numbers values) found
  cases values with
  | nil => cases nonempty rfl
  | cons first rest =>
      simpa [Mutable.allocateTuple, numbers, Except.map] using converted

/-- Flush and high-card returns allocate the copied values tuple before their outer rank tuple. -/
private theorem nested_rank (heap : Heap) (env : HeapCalls.Env) (address : Nat) (values : List Int)
    (category : Int) (globals : Mutable.Globals) (nonempty : values ≠ [])
    (bound : env "values" = some (.ref address)) (found : heap[address]? = some (.list (numbers values))) :
    Evaluates program env (.tuple [.constant (.int category), .builtinCall .tuple [.name "values"]]) heap
      (.ref (heap.length+1))
      (heap ++ [.tuple (numbers values), .tuple [number category, .ref heap.length]]) globals := by
  have constructed : Evaluates program env (.tuple [.constant (.int category), .builtinCall .tuple [.name "values"]]) heap
      (.ref (heap ++ [Mutable.Object.tuple (numbers values)]).length)
      ((heap ++ [.tuple (numbers values)]) ++ [.tuple [number category, .ref heap.length]]) globals :=
    Evaluates.tuple (.cons Evaluates.constant (.cons (values_tuple heap env address values globals nonempty bound found) .nil))
      (by simp)
  simpa only [List.length_append, List.length_singleton, List.append_assoc] using constructed

def answer (heap : Heap) (flush straight : Bool) : Mutable.Value :=
  if flush then .ref (heap.length+1) else if straight then .ref heap.length else .ref (heap.length+1)

def output (heap : Heap) (values : List Int) (flush straight : Bool) (high : Int) : Heap :=
  if flush then heap ++ [.tuple (numbers values), .tuple [number 5, .ref heap.length]]
  else if straight then heap ++ [.tuple [number 4, number high]]
  else heap ++ [.tuple (numbers values), .tuple [number 0, .ref heap.length]]

/-- The actual no-pair/no-triple/no-quad suffix returns flush, straight, or high-card ranks. -/
theorem result (heap : Heap) (env : HeapCalls.Env) (valuesAddress groupsAddress firstAddress : Nat)
    (values : List Int) (groups : List Mutable.Value) (count rank high : Int) (flush straight : Bool)
    (globals : Mutable.Globals) (nonempty : values ≠ [])
    (getValues : env "values" = some (.ref valuesAddress))
    (valuesFound : heap[valuesAddress]? = some (.list (numbers values)))
    (getGroups : env "groups" = some (.ref groupsAddress))
    (groupsFound : heap[groupsAddress]? = some (.list groups))
    (first : groups[0]? = some (.ref firstAddress))
    (pair : heap[firstAddress]? = some (.tuple [number count, number rank]))
    (getFlush : env "is_flush" = some (flag flush))
    (getStraight : env "is_straight" = some (flag straight))
    (getHigh : env "straight_high" = some (number high))
    (notTwo : count ≠ 2) (notThree : count ≠ 3) (notFour : count ≠ 4) :
    BlockReturns program env (code.drop 9) heap (answer heap flush straight)
      (output heap values flush straight high) globals := by
  have counted := straight_detection.group_count heap env groupsAddress 0 firstAddress groups count rank globals
    getGroups groupsFound first pair
  have noFour : Evaluates program env
      (.compare (straight_detection.groupExpression 0 0) [(.eq, .constant (.int 4))]) heap (flag false) heap globals := by
    simpa only [notFour, decide_false] using
      equal_integer heap env (straight_detection.groupExpression 0 0) count 4 globals counted
  have noThree : Evaluates program env
      (.compare (straight_detection.groupExpression 0 0) [(.eq, .constant (.int 3))]) heap (flag false) heap globals := by
    simpa only [notThree, decide_false] using
      equal_integer heap env (straight_detection.groupExpression 0 0) count 3 globals counted
  have noTwo : Evaluates program env
      (.compare (straight_detection.groupExpression 0 0) [(.eq, .constant (.int 2))]) heap (flag false) heap globals := by
    simpa only [notTwo, decide_false] using
      equal_integer heap env (straight_detection.groupExpression 0 0) count 2 globals counted
  apply BlockReturns.branch (flag := false) (value := flag false) (middle := heap) noFour rfl
  apply BlockReturns.branch (flag := false) (value := flag false) (middle := heap)
    (Evaluates.boolean_stop (flag := false) noThree rfl rfl) rfl
  cases flush with
  | true =>
      simp only [answer, output, ↓reduceIte]
      apply BlockReturns.branch (flag := true) (value := flag true) (middle := heap) (Evaluates.lookup getFlush) rfl
      exact BlockReturns.return_value (nested_rank heap env valuesAddress values 5 globals nonempty getValues valuesFound)
  | false =>
      apply BlockReturns.branch (flag := false) (value := flag false) (middle := heap) (Evaluates.lookup getFlush) rfl
      cases straight with
      | true =>
          simp only [answer, output, ↓reduceIte]
          apply BlockReturns.branch (flag := true) (value := flag true) (middle := heap) (Evaluates.lookup getStraight) rfl
          apply BlockReturns.return_value
          exact Evaluates.tuple (.cons Evaluates.constant (.cons (Evaluates.lookup getHigh) .nil)) (by simp)
      | false =>
          simp only [answer, output]
          apply BlockReturns.branch (flag := false) (value := flag false) (middle := heap) (Evaluates.lookup getStraight) rfl
          apply BlockReturns.branch (flag := false) (value := flag false) (middle := heap) noThree rfl
          apply BlockReturns.branch (flag := false) (value := flag false) (middle := heap)
            (Evaluates.boolean_stop (flag := false) noTwo rfl rfl) rfl
          apply BlockReturns.branch (flag := false) (value := flag false) (middle := heap) noTwo rfl
          exact BlockReturns.return_value (nested_rank heap env valuesAddress values 0 globals nonempty getValues valuesFound)

end solvers.«54».numeric_score.simple_categories

namespace solvers.«54».numeric_score.repeated_categories
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
set_option Elab.async false

private abbrev program := Solvers.P54Ast.module
private abbrev number (n : Int) : Mutable.Value := .scalar (.int n)
private abbrev numbers (values : List Int) : List Mutable.Value := values.map number
private abbrev flag (b : Bool) : Mutable.Value := .scalar (.bool b)
private def code : List Stmt := executableBody ((findFunctionIn program.body "_score_values").getD ([], [])).2

/-- The actual full-house suffix skips the quad test, reads both source pairs, and returns its rank tuple. -/
theorem full_house (heap : Heap) (env : HeapCalls.Env) (groupsAddress firstAddress secondAddress : Nat)
    (groups : List Mutable.Value) (trip pair : Int) (globals : Mutable.Globals)
    (getGroups : env "groups" = some (.ref groupsAddress))
    (groupsFound : heap[groupsAddress]? = some (.list groups))
    (first : groups[0]? = some (.ref firstAddress)) (second : groups[1]? = some (.ref secondAddress))
    (tripFound : heap[firstAddress]? = some (.tuple [number 3, number trip]))
    (pairFound : heap[secondAddress]? = some (.tuple [number 2, number pair])) :
    BlockReturns program env (code.drop 9) heap (.ref heap.length)
      (heap ++ [.tuple [number 6, number trip, number pair]]) globals := by
  apply BlockReturns.branch (flag := false) (value := flag false) (middle := heap)
  · exact Evaluates.compare
      (straight_detection.group_count heap env groupsAddress 0 firstAddress groups 3 trip globals getGroups groupsFound first tripFound)
      Evaluates.constant rfl
  · rfl
  · apply BlockReturns.branch (flag := true) (value := flag true) (middle := heap)
    · exact Evaluates.boolean_next (flag := true)
        (Evaluates.compare
          (straight_detection.group_count heap env groupsAddress 0 firstAddress groups 3 trip globals getGroups groupsFound first tripFound)
          Evaluates.constant rfl) rfl rfl
        (Evaluates.boolean_last (flag := true)
          (Evaluates.compare
            (straight_detection.group_count heap env groupsAddress 1 secondAddress groups 2 pair globals getGroups groupsFound second pairFound)
            Evaluates.constant rfl) rfl)
    · rfl
    · apply BlockReturns.assign
        (straight_detection.group_rank heap env groupsAddress 0 firstAddress groups 3 trip globals getGroups groupsFound first tripFound)
      apply BlockReturns.assign
        (straight_detection.group_rank heap (setLocal env "trip_val" (number trip)) groupsAddress 1 secondAddress groups 2 pair globals
          (by simpa [setLocal] using getGroups) groupsFound second pairFound)
      apply BlockReturns.return_value
      exact Evaluates.tuple (.cons Evaluates.constant
        (.cons (Evaluates.lookup (by simp [setLocal])) (.cons (Evaluates.lookup rfl) .nil))) (by simp)

end solvers.«54».numeric_score.repeated_categories

namespace solvers.«54».numeric_score.filtered_categories
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
set_option Elab.async false

private abbrev program := Solvers.P54Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
private abbrev numbers (values : List Int) : List Mutable.Value := values.map number
private abbrev flag (value : Bool) : Mutable.Value := .scalar (.bool value)
private def code : List Stmt := executableBody ((findFunctionIn program.body "_score_values").getD ([], [])).2

def filtered (values : List Int) (rank : Int) : List Int := values.filter (fun value => value != rank)
def kickers (values : List Int) (rank : Int) : List Int := sortedInts true (filtered values rank)
private def collector (heap : Heap) (values : List Int) : Heap := heap ++ [.list (numbers values)]
def sortedHeap (heap : Heap) (values : List Int) (rank : Int) : Heap :=
  heap ++ [.list (numbers (filtered values rank)), .list (numbers (kickers values rank))]
def output (heap : Heap) (values : List Int) (rank category : Int) : Heap :=
  heap ++ [.list (numbers (filtered values rank)), .list (numbers (kickers values rank)),
    .tuple (numbers (kickers values rank)), .tuple [number category, number rank, .ref (heap.length+2)]]

private def predicate (name : String) : Expr := .compare (.name "v") [(.notEq, .name name)]
private def generators (name : String) : List (Expr × Expr × List Expr) :=
  [(.name "v", .name "values", [predicate name])]
private def filterBody (name : String) (address : Nat) : List Stmt :=
  [.ifThen (predicate name) [.comprehensionAppend address (.name "v")] []]
private def sortedExpression (name : String) : Expr :=
  .consumeGenerator (.builtinCall .sortedReverse [.constant (.bool true)]) (.name "v") (generators name)

private theorem test (heap : Heap) (env : HeapCalls.Env) (name : String) (value rank : Int)
    (globals : Mutable.Globals) (getValue : env "v" = some (number value)) (getRank : env name = some (number rank)) :
    Evaluates program env (predicate name) heap (flag (decide (value ≠ rank))) heap globals := by
  apply Evaluates.compare (av := number value) (bv := number rank) (middle := heap) (afterRight := heap)
    (Evaluates.lookup getValue) (Evaluates.lookup getRank)
  simp [HeapCalls.compare, scalarCompare, evalCmpOp, flag]

private theorem collect_loop (heap : Heap) (address : Nat) (values : List Int)
    (found : heap[address]? = some (.list (numbers values))) (name : String) (rank : Int) (notV : name ≠ "v")
    (remaining before accumulated : List Int) (decomposition : values = before ++ remaining)
    (env : HeapCalls.Env) (globals : Mutable.Globals) (getRank : env name = some (number rank)) :
    ∃ finalEnv, BlockContinues program env
      [.sequenceStepLoop "v" (.ref address) before.length (filterBody name heap.length) []]
      (collector heap accumulated) finalEnv (collector heap (accumulated ++ filtered remaining rank)) globals := by
  have sequence (collected : List Int) : Mutable.sequence (.ref address) (collector heap collected) =
      some (.ok (numbers values), collector heap collected) := by
    have original := ProjectEulerSolutions.HeapCollection.prefix_found heap [.list (numbers collected)] address _ found
    simp [Mutable.sequence, original, collector]
  induction remaining generalizing before accumulated env with
  | nil =>
      refine ⟨env, ?_⟩
      apply BlockContinues.sequence_done (source := .ref address) (items := numbers values) (sequence accumulated)
      · simp [numbers, decomposition]
      · simpa only [filtered, List.filter_nil, List.append_nil] using
          (BlockContinues.nil (module := program) (env := env) (heap := collector heap accumulated) (globals := globals))
  | cons value rest ih =>
      let bound := setLocal env "v" (number value)
      have getBound : bound name = some (number rank) := by simpa [bound, setLocal, notV] using getRank
      have nextDecomposition : values = (before ++ [value]) ++ rest := by
        simpa only [List.append_assoc, List.singleton_append] using decomposition
      have checked := test (collector heap accumulated) bound name value rank globals rfl getBound
      by_cases same : value = rank
      · have skipped : BlockContinues program bound (filterBody name heap.length)
            (collector heap accumulated) bound (collector heap accumulated) globals := by
          apply BlockContinues.branch (flag := false) (value := flag false) (middle := collector heap accumulated)
          · simpa only [same, ne_eq, not_true_eq_false, decide_false] using checked
          · rfl
          · exact BlockContinues.nil
        obtain ⟨finalEnv, continued⟩ := ih (before ++ [value]) accumulated nextDecomposition bound getBound
        refine ⟨finalEnv, ?_⟩
        apply BlockContinues.sequence_step (source := .ref address) (items := numbers values) (value := number value) (sequence accumulated)
        · simp [numbers, decomposition]
        · apply BlockContinues.append skipped
          change BlockContinues program bound
            [.sequenceStepLoop "v" (.ref address) (before.length+1) (filterBody name heap.length) []]
            (collector heap accumulated) finalEnv (collector heap (accumulated ++ filtered (value :: rest) rank)) globals
          simpa [filtered, same] using continued
      · have emitted : BlockContinues program bound [.comprehensionAppend heap.length (.name "v")]
            (collector heap accumulated) bound (collector heap (accumulated ++ [value])) globals := by
          have written := BlockContinues.comprehension_append (module := program) (env := bound) (globals := globals)
            (heap := collector heap accumulated) (address := heap.length) (items := numbers accumulated)
            (Evaluates.lookup (name := "v") (value := number value) rfl)
            (by simp [collector])
          simpa [collector, numbers, List.map_append, List.set_append_right] using written
        have kept : BlockContinues program bound (filterBody name heap.length)
            (collector heap accumulated) bound (collector heap (accumulated ++ [value])) globals := by
          apply BlockContinues.branch (flag := true) (value := flag true) (middle := collector heap accumulated)
          · simpa only [same, ne_eq, not_false_eq_true, decide_true] using checked
          · rfl
          · exact emitted
        obtain ⟨finalEnv, continued⟩ := ih (before ++ [value]) (accumulated ++ [value]) nextDecomposition bound getBound
        refine ⟨finalEnv, ?_⟩
        apply BlockContinues.sequence_step (source := .ref address) (items := numbers values) (value := number value) (sequence accumulated)
        · simp [numbers, decomposition]
        · apply BlockContinues.append kept
          change BlockContinues program bound
            [.sequenceStepLoop "v" (.ref address) (before.length+1) (filterBody name heap.length) []]
            (collector heap (accumulated ++ [value])) finalEnv
            (collector heap (accumulated ++ filtered (value :: rest) rank)) globals
          simpa [filtered, same, List.append_assoc] using continued

/-- The real generator filters first and then performs stable reverse sorting. -/
theorem sorted_filter (heap : Heap) (env : HeapCalls.Env) (address : Nat) (values : List Int)
    (name : String) (rank : Int) (globals : Mutable.Globals) (notV : name ≠ "v")
    (getValues : env "values" = some (.ref address)) (getRank : env name = some (number rank))
    (found : heap[address]? = some (.list (numbers values))) :
    Evaluates program env (sortedExpression name) heap (.ref (heap.length+1)) (sortedHeap heap values rank) globals := by
  let inside := comprehensionTargetsEnv env (generators name)
  have getInside : inside name = some (number rank) := by
    simpa [inside, generators, comprehensionTargetsEnv, targetLocals, notV, Ne.symm notV] using getRank
  obtain ⟨finalEnv, collected⟩ := collect_loop heap address values found name rank notV values [] [] (by simp) inside globals getInside
  have ordered : HeapSort.sortWithReverse (collector heap (filtered values rank)) true (numbers (filtered values rank)) =
      some (.ok (numbers (kickers values rank))) := by
    simpa only [numbers, List.map_map, Function.comp_def] using
      sortWithReverse_ints (collector heap (filtered values rank)) true (filtered values rank)
  have finish := primitive_sorted_reverse_list (collector heap (filtered values rank)) heap.length
    (numbers (filtered values rank)) (numbers (kickers values rank)) true (by simp [collector]) ordered
  apply Evaluates.consume_generator_sorted_reverse
  apply GeneratorCollects.sequence (target := .name "v") (iterable := .name "values")
    (filters := [predicate name]) (generators := []) (finalEnv := finalEnv) (items := numbers values)
    (source := .ref address) (middle := heap) (collectedHeap := collector heap (filtered values rank))
    (by intro expressions impossible; cases impossible) (by intro expressions impossible; cases impossible)
    (by intro expressions impossible; cases impossible) (Evaluates.lookup getValues) (by simp [Mutable.sequence, found])
  · simpa only [List.nil_append, generators, filterBody, comprehensionTargetName, comprehensionTargetBody,
      comprehensionFilters, comprehensionTargetsBody, List.foldr_cons, List.foldr_nil] using collected
  · change primitive (.builtin .sortedReverse) [.ref heap.length, .scalar (.bool true)] _ = _
    simpa only [collector, sortedHeap, List.length_append, List.length_singleton, List.append_assoc] using finish

theorem nonempty_of_count_lt (values : List Int) (rank : Int) (less : values.count rank < values.length) :
    filtered values rank ≠ [] := by
  intro empty
  have partition := solvers.«54».Kickers.length_partition values rank
  change values.filter (fun value => value != rank) = [] at empty
  rw [empty] at partition
  simp only [List.length_nil] at partition
  omega

theorem nonempty_three (values : List Int) (rank : Int) (five : values.length = 5) (three : values.count rank = 3) :
    filtered values rank ≠ [] := nonempty_of_count_lt values rank (by omega)

theorem nonempty_pair (values : List Int) (rank : Int) (five : values.length = 5) (two : values.count rank = 2) :
    filtered values rank ≠ [] := nonempty_of_count_lt values rank (by omega)

private theorem sorted_nonempty (values : List Int) (rank : Int) (nonempty : filtered values rank ≠ []) :
    kickers values rank ≠ [] := by
  have sameLength : (kickers values rank).length = (filtered values rank).length := by
    simpa only [kickers, sortedInts, ↓reduceIte, List.length_reverse] using
      (ScalarSort.ints_perm (filtered values rank).reverse).length_eq
  intro empty
  apply nonempty
  apply List.length_eq_zero_iff.mp
  simpa only [empty, List.length_nil] using sameLength.symm

private theorem copy_tuple (heap : Heap) (env : HeapCalls.Env) (address : Nat) (values : List Int)
    (globals : Mutable.Globals) (nonempty : values ≠ [])
    (bound : env "kickers" = some (.ref address)) (found : heap[address]? = some (.list (numbers values))) :
    Evaluates program env (.builtinCall .tuple [.name "kickers"]) heap (.ref heap.length)
      (heap ++ [.tuple (numbers values)]) globals := by
  apply Evaluates.builtin (.cons (Evaluates.lookup bound) .nil)
  have converted := primitive_tuple_list heap address (numbers values) found
  cases values with
  | nil => cases nonempty rfl
  | cons first rest => simpa [Mutable.allocateTuple, numbers, Except.map] using converted

private def tailBody (name : String) (category : Int) : List Stmt :=
  [.assign [.name name] (straight_detection.groupExpression 0 1),
   .assign [.name "kickers"] (sortedExpression name),
   .return (.tuple [.constant (.int category), .name name, .builtinCall .tuple [.name "kickers"]])]

private theorem tail_result (heap : Heap) (env : HeapCalls.Env) (valuesAddress groupsAddress firstAddress : Nat)
    (values : List Int) (groups : List Mutable.Value) (frequency rank category : Int)
    (name : String) (globals : Mutable.Globals) (notV : name ≠ "v") (notValues : name ≠ "values") (notKickers : name ≠ "kickers")
    (nonempty : filtered values rank ≠ []) (getValues : env "values" = some (.ref valuesAddress))
    (valuesFound : heap[valuesAddress]? = some (.list (numbers values)))
    (getGroups : env "groups" = some (.ref groupsAddress)) (groupsFound : heap[groupsAddress]? = some (.list groups))
    (first : groups[0]? = some (.ref firstAddress)) (pair : heap[firstAddress]? = some (.tuple [number frequency, number rank]))
    (following : List Stmt) :
    BlockReturns program env (tailBody name category ++ following) heap (.ref (heap.length+3)) (output heap values rank category) globals := by
  let named := setLocal env name (number rank)
  let ready := setLocal named "kickers" (.ref (heap.length+1))
  have getNamedValues : named "values" = some (.ref valuesAddress) := by
    simpa [named, setLocal, Ne.symm notValues] using getValues
  have sorted := sorted_filter heap named valuesAddress values name rank globals notV getNamedValues
    (by simp [named, setLocal]) valuesFound
  apply BlockReturns.assign (straight_detection.group_rank heap env groupsAddress 0 firstAddress groups frequency rank globals
    getGroups groupsFound first pair)
  apply BlockReturns.assign sorted
  apply BlockReturns.return_value
  have getRank : ready name = some (number rank) := by simp [ready, named, setLocal, notKickers]
  have foundKickers : (sortedHeap heap values rank)[heap.length+1]? = some (.list (numbers (kickers values rank))) := by
    simp [sortedHeap]
  have copied := copy_tuple (sortedHeap heap values rank) ready (heap.length+1) (kickers values rank) globals
    (sorted_nonempty values rank nonempty) rfl foundKickers
  have constructed : Evaluates program ready
      (.tuple [.constant (.int category), .name name, .builtinCall .tuple [.name "kickers"]])
      (sortedHeap heap values rank) (.ref (sortedHeap heap values rank ++ [Mutable.Object.tuple (numbers (kickers values rank))]).length)
      ((sortedHeap heap values rank ++ [.tuple (numbers (kickers values rank))]) ++
        [.tuple [number category, number rank, .ref (sortedHeap heap values rank).length]]) globals :=
    Evaluates.tuple (.cons Evaluates.constant (.cons (Evaluates.lookup getRank) (.cons copied .nil))) (by simp)
  simpa only [output, sortedHeap, List.length_append, List.length_cons, List.length_nil, Nat.zero_add,
    List.append_assoc, List.cons_append, List.nil_append] using constructed

private theorem eq_test (heap : Heap) (env : HeapCalls.Env) (expression : Expr) (value wanted : Int)
    (globals : Mutable.Globals) (evaluated : Evaluates program env expression heap (number value) heap globals) :
    Evaluates program env (.compare expression [(.eq, .constant (.int wanted))]) heap
      (flag (decide (value = wanted))) heap globals := by
  apply Evaluates.compare (av := number value) (bv := number wanted) (middle := heap) (afterRight := heap) evaluated Evaluates.constant
  simp [HeapCalls.compare, scalarCompare, evalCmpOp, flag]

/-- The non-full-house triple path executes its actual filtered generator and returns category three. -/
theorem trips (heap : Heap) (env : HeapCalls.Env) (valuesAddress groupsAddress firstAddress secondAddress : Nat)
    (values : List Int) (groups : List Mutable.Value) (rank secondCount secondRank : Int) (globals : Mutable.Globals)
    (getValues : env "values" = some (.ref valuesAddress)) (valuesFound : heap[valuesAddress]? = some (.list (numbers values)))
    (getGroups : env "groups" = some (.ref groupsAddress)) (groupsFound : heap[groupsAddress]? = some (.list groups))
    (first : groups[0]? = some (.ref firstAddress)) (second : groups[1]? = some (.ref secondAddress))
    (firstPair : heap[firstAddress]? = some (.tuple [number 3, number rank]))
    (secondPair : heap[secondAddress]? = some (.tuple [number secondCount, number secondRank]))
    (notFull : secondCount ≠ 2) (getFlush : env "is_flush" = some (flag false))
    (getStraight : env "is_straight" = some (flag false)) (nonempty : filtered values rank ≠ []) :
    BlockReturns program env (code.drop 9) heap (.ref (heap.length+3)) (output heap values rank 3) globals := by
  have firstCount := straight_detection.group_count heap env groupsAddress 0 firstAddress groups 3 rank globals getGroups groupsFound first firstPair
  have secondCounted := straight_detection.group_count heap env groupsAddress 1 secondAddress groups secondCount secondRank globals
    getGroups groupsFound second secondPair
  have noFour := eq_test heap env _ 3 4 globals firstCount
  have yesThree := eq_test heap env _ 3 3 globals firstCount
  have noSecondTwo := eq_test heap env _ secondCount 2 globals secondCounted
  apply BlockReturns.branch (flag := false) (value := flag false) (middle := heap)
    (by simpa using noFour) rfl
  apply BlockReturns.branch (flag := false) (value := flag false) (middle := heap)
    (Evaluates.boolean_next (flag := true) (by simpa using yesThree) rfl rfl
      (Evaluates.boolean_last (flag := false) (by simpa [notFull] using noSecondTwo) rfl)) rfl
  apply BlockReturns.branch (flag := false) (value := flag false) (middle := heap) (Evaluates.lookup getFlush) rfl
  apply BlockReturns.branch (flag := false) (value := flag false) (middle := heap) (Evaluates.lookup getStraight) rfl
  apply BlockReturns.branch (flag := true) (value := flag true) (middle := heap) (by simpa using yesThree) rfl
  exact tail_result heap env valuesAddress groupsAddress firstAddress values groups 3 rank 3 "trip_val" globals
    (by decide) (by decide) (by decide) nonempty getValues valuesFound getGroups groupsFound first firstPair (code.drop 14)

/-- A single-pair profile skips the two-pair branch before constructing its real tuple tail. -/
theorem pair (heap : Heap) (env : HeapCalls.Env) (valuesAddress groupsAddress firstAddress secondAddress : Nat)
    (values : List Int) (groups : List Mutable.Value) (rank secondCount secondRank : Int) (globals : Mutable.Globals)
    (getValues : env "values" = some (.ref valuesAddress)) (valuesFound : heap[valuesAddress]? = some (.list (numbers values)))
    (getGroups : env "groups" = some (.ref groupsAddress)) (groupsFound : heap[groupsAddress]? = some (.list groups))
    (first : groups[0]? = some (.ref firstAddress)) (second : groups[1]? = some (.ref secondAddress))
    (firstPair : heap[firstAddress]? = some (.tuple [number 2, number rank]))
    (secondPair : heap[secondAddress]? = some (.tuple [number secondCount, number secondRank]))
    (notTwoPairs : secondCount ≠ 2) (getFlush : env "is_flush" = some (flag false))
    (getStraight : env "is_straight" = some (flag false)) (nonempty : filtered values rank ≠ []) :
    BlockReturns program env (code.drop 9) heap (.ref (heap.length+3)) (output heap values rank 1) globals := by
  have firstCount := straight_detection.group_count heap env groupsAddress 0 firstAddress groups 2 rank globals getGroups groupsFound first firstPair
  have secondCounted := straight_detection.group_count heap env groupsAddress 1 secondAddress groups secondCount secondRank globals
    getGroups groupsFound second secondPair
  have noFour := eq_test heap env _ 2 4 globals firstCount
  have noThree := eq_test heap env _ 2 3 globals firstCount
  have yesTwo := eq_test heap env _ 2 2 globals firstCount
  have noSecondTwo := eq_test heap env _ secondCount 2 globals secondCounted
  apply BlockReturns.branch (flag := false) (value := flag false) (middle := heap) (by simpa using noFour) rfl
  apply BlockReturns.branch (flag := false) (value := flag false) (middle := heap)
    (Evaluates.boolean_stop (flag := false) (by simpa using noThree) rfl rfl) rfl
  apply BlockReturns.branch (flag := false) (value := flag false) (middle := heap) (Evaluates.lookup getFlush) rfl
  apply BlockReturns.branch (flag := false) (value := flag false) (middle := heap) (Evaluates.lookup getStraight) rfl
  apply BlockReturns.branch (flag := false) (value := flag false) (middle := heap) (by simpa using noThree) rfl
  apply BlockReturns.branch (flag := false) (value := flag false) (middle := heap)
    (Evaluates.boolean_next (flag := true) (by simpa using yesTwo) rfl rfl
      (Evaluates.boolean_last (flag := false) (by simpa [notTwoPairs] using noSecondTwo) rfl)) rfl
  apply BlockReturns.branch (flag := true) (value := flag true) (middle := heap) (by simpa using yesTwo) rfl
  exact tail_result heap env valuesAddress groupsAddress firstAddress values groups 2 rank 1 "pair_val" globals
    (by decide) (by decide) (by decide) nonempty getValues valuesFound getGroups groupsFound first firstPair (code.drop 16)

end solvers.«54».numeric_score.filtered_categories

namespace solvers.«54».numeric_score.composition
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
set_option Elab.async false

private abbrev program := Solvers.P54Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
private abbrev numbers (values : List Int) : List Mutable.Value := values.map number
private abbrev flag (value : Bool) : Mutable.Value := .scalar (.bool value)
private def code : List Stmt := executableBody ((findFunctionIn program.body "_score_values").getD ([], [])).2
private abbrev sourceArgs := ((findFunctionIn program.body "_score_values").getD ([], [])).1
private abbrev sourceBody := ((findFunctionIn program.body "_score_values").getD ([], [])).2

def entry (globals : Mutable.Globals) (valuesAddress suitsAddress : Nat) : HeapCalls.Env :=
  functionEnv sourceArgs sourceBody [.ref valuesAddress, .ref suitsAddress] globals

def preparedHeap (heap : Heap) (valuesAddress : Nat) (values : List Int) (suits : List String) : Heap :=
  normalizationHeap heap valuesAddress values suits ++ straight_detection.extra (orderedUnique values)

def preparedEnv (env : HeapCalls.Env) (heap : Heap) (values : List Int) (suits : List String) : HeapCalls.Env :=
  straight_detection.updated (normalizationEnv env heap suits) (orderedUnique values)

def scoredHeap (heap : Heap) (valuesAddress : Nat) (values : List Int) (suits : List String) : Heap :=
  straight_detection.groupedHeap (preparedHeap heap valuesAddress values suits) (descending values)

def scoredEnv (env : HeapCalls.Env) (heap : Heap) (valuesAddress : Nat) (values : List Int)
    (suits : List String) : HeapCalls.Env :=
  straight_detection.groupedEnv (preparedEnv env heap values suits)
    (preparedHeap heap valuesAddress values suits) (descending values)

theorem prepared_values (heap : Heap) (address : Nat) (values : List Int) (suits : List String)
    (found : heap[address]? = some (.list (numbers values))) :
    (preparedHeap heap address values suits)[address]? = some (.list (numbers (descending values))) := by
  have inside := (List.getElem?_eq_some_iff.mp found).1
  have stored : (heap.set address (.list (numbers (descending values))))[address]? =
      some (.list (numbers (descending values))) := by simp [inside]
  have first := ProjectEulerSolutions.HeapCollection.prefix_found
    (heap.set address (.list (numbers (descending values))))
    [.set (suitKeys suits), .set ((unique (descending values)).map Constant.int), .list (numbers (orderedUnique values))]
    address _ stored
  exact ProjectEulerSolutions.HeapCollection.prefix_found _ _ address _ first

theorem prepared_other (env : HeapCalls.Env) (heap : Heap) (values : List Int) (suits : List String)
    (name : String) (notFlush : name ≠ "is_flush") (notUnique : name ≠ "uniq")
    (notStraight : name ≠ "is_straight") (notHigh : name ≠ "straight_high") :
    preparedEnv env heap values suits name = env name := by
  rw [preparedEnv, straight_detection.updated_other _ _ name notStraight notHigh]
  simp [normalizationEnv, setLocal, notFlush, notUnique, notStraight, notHigh]

theorem prepared_straight (env : HeapCalls.Env) (heap : Heap) (values : List Int) (suits : List String) :
    preparedEnv env heap values suits "is_straight" = some (flag (straight_detection.detected (orderedUnique values))) :=
  straight_detection.updated_straight _ _ (by simp [normalizationEnv, setLocal])

theorem prepared_flush (env : HeapCalls.Env) (heap : Heap) (values : List Int) (suits : List String) :
    preparedEnv env heap values suits "is_flush" = some (flag (isFlush suits)) := by
  rw [preparedEnv, straight_detection.updated_other _ _ "is_flush" (by decide) (by decide)]
  simp [normalizationEnv, setLocal]

/-- The real source prefix builds the exact groups used by every remaining category. -/
theorem through_groups (heap : Heap) (env : HeapCalls.Env) (valuesAddress suitsAddress : Nat)
    (values : List Int) (suits : List String) (globals : Mutable.Globals)
    (nonempty : values ≠ [])
    (valuesFound : heap[valuesAddress]? = some (.list (numbers values)))
    (suitsFound : heap[suitsAddress]? = some (.list (suits.map fun value => .scalar (.string value))))
    (getValues : env "values" = some (.ref valuesAddress)) (getSuits : env "suits" = some (.ref suitsAddress))
    (notBoth : (straight_detection.detected (orderedUnique values) && isFlush suits) = false) :
    BlockContinues program env (code.take 9) heap (scoredEnv env heap valuesAddress values suits)
      (scoredHeap heap valuesAddress values suits) globals := by
  have normalized := straight_detection.normalization_straight heap env valuesAddress suitsAddress values suits globals
    nonempty valuesFound suitsFound getValues getSuits
  have grouped := straight_detection.groups_phase (preparedHeap heap valuesAddress values suits)
    (preparedEnv env heap values suits) valuesAddress (descending values)
    (straight_detection.detected (orderedUnique values)) (isFlush suits) globals
    (prepared_values heap valuesAddress values suits valuesFound)
    (by simpa only [prepared_other env heap values suits "values" (by decide) (by decide) (by decide) (by decide)] using getValues)
    (prepared_straight env heap values suits) (prepared_flush env heap values suits) notBoth
  simpa [code, scoredEnv, scoredHeap, preparedEnv, preparedHeap] using normalized.append grouped

/-- A proved category suffix closes the whole numeric helper without any assumed callee execution. -/
theorem result_of_suffix (heap : Heap) (valuesAddress suitsAddress : Nat)
    (values : List Int) (suits : List String) (globals : Mutable.Globals)
    (nonempty : values ≠ [])
    (valuesFound : heap[valuesAddress]? = some (.list (numbers values)))
    (suitsFound : heap[suitsAddress]? = some (.list (suits.map fun value => .scalar (.string value))))
    (notBoth : (straight_detection.detected (orderedUnique values) && isFlush suits) = false)
    (answer : Mutable.Value) (finalHeap : Heap)
    (suffix : BlockReturns program (scoredEnv (entry globals valuesAddress suitsAddress) heap valuesAddress values suits)
      (code.drop 9) (scoredHeap heap valuesAddress values suits) answer finalHeap globals) :
    ReturnsValue program "_score_values" [.ref valuesAddress, .ref suitsAddress] heap answer finalHeap globals := by
  have getValues : entry globals valuesAddress suitsAddress "values" = some (.ref valuesAddress) := by
    cases globals <;> rfl
  have getSuits : entry globals valuesAddress suitsAddress "suits" = some (.ref suitsAddress) := by
    cases globals <;> rfl
  have prefixRun := through_groups heap (entry globals valuesAddress suitsAddress) valuesAddress suitsAddress values suits globals
    nonempty valuesFound suitsFound getValues getSuits notBoth
  apply BlockReturns.function rfl rfl
  simpa only [List.take_append_drop] using prefixRun.returns suffix

theorem scored_profile (heap : Heap) (valuesAddress : Nat) (values : List Int) (suits : List String)
    (five : values.length = 5) :
    ((rank_groups.resultGroups (straight_detection.countedHeap (preparedHeap heap valuesAddress values suits) (descending values))
      (count_values.counts (descending values))).map
      (ProjectEulerSolutions.TupleSorting.readInts (scoredHeap heap valuesAddress values suits))).map (fun row => row.headD 0) ∈
        ProjectEulerSolutions.PokerCountPatterns.profiles := by
  have length : (descending values).length = 5 :=
    (ProjectEulerSolutions.PokerUnique.descending_perm values).length_eq.trans five
  exact straight_detection.grouped_profile _ _ length

end solvers.«54».numeric_score.composition

namespace solvers.«54».numeric_score.repeated_categories.quad_path
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
set_option Elab.async false

private abbrev program := Solvers.P54Ast.module
private abbrev number (n : Int) : Mutable.Value := .scalar (.int n)
private abbrev numbers (values : List Int) : List Mutable.Value := values.map number
private abbrev flag (b : Bool) : Mutable.Value := .scalar (.bool b)
private def code : List Stmt := executableBody ((findFunctionIn program.body "_score_values").getD ([], [])).2

def Binds (env : HeapCalls.Env) (captures : List (String × Mutable.Value)) : Prop :=
  ∀ capture ∈ captures, env capture.1 = some capture.2

private abbrev working (heap : Heap) (collected : List Int) : Heap := heap ++ [.list (numbers collected)]
private abbrev filterBody (address : Nat) (test : Expr) : List Stmt :=
  [.ifThen test [.comprehensionAppend address (.name "v")] []]

private theorem binds_v (env : HeapCalls.Env) (captures : List (String × Mutable.Value)) (value : Int)
    (bound : Binds env captures) (noShadow : ∀ capture ∈ captures, capture.1 ≠ "v") :
    Binds (setLocal env "v" (number value)) captures := by
  intro capture member
  simpa [setLocal, noShadow capture member] using bound capture member

private theorem binds_comprehension (env : HeapCalls.Env) (captures : List (String × Mutable.Value)) (test : Expr)
    (bound : Binds env captures) (noShadow : ∀ capture ∈ captures, capture.1 ≠ "v") :
    Binds (comprehensionEnv env [("v", .name "values", [test])]) captures := by
  intro capture member
  simpa [comprehensionEnv, Ne.symm (noShadow capture member)] using bound capture member

/-- The private comprehension collector reads the unchanged source list and preserves captured bindings. -/
theorem collect_filter (heap : Heap) (address : Nat) (values remaining : List Int)
    (found : heap[address]? = some (.list (numbers values))) (position : Nat) (suffix : values.drop position = remaining)
    (collected : List Int) (env : HeapCalls.Env) (test : Expr) (accept : Int → Bool)
    (captures : List (String × Mutable.Value)) (globals : Mutable.Globals)
    (noShadow : ∀ capture ∈ captures, capture.1 ≠ "v")
    (checked : ∀ (collected : List Int) (inner : HeapCalls.Env) (value : Int), Binds inner captures →
      inner "v" = some (number value) →
      Evaluates program inner test (working heap collected) (flag (accept value)) (working heap collected) globals)
    (bound : Binds env captures) :
    ∃ finalEnv, BlockContinues program env
      [.sequenceStepLoop "v" (.ref address) position (filterBody heap.length test) []]
      (working heap collected) finalEnv (working heap (collected ++ remaining.filter accept)) globals := by
  have inside := (List.getElem?_eq_some_iff.mp found).1
  have sequence (items : List Int) : Mutable.sequence (.ref address) (working heap items) =
      some (.ok (numbers values), working heap items) := by
    simp [working, Mutable.sequence, List.getElem?_append_left inside, found]
  induction remaining generalizing position collected env with
  | nil =>
      simp only [List.filter_nil, List.append_nil]
      refine ⟨env, BlockContinues.sequence_done (source := .ref address) (sequence collected) ?_ .nil⟩
      have ended : values[position]? = none := by
        simpa [suffix] using (List.head?_drop (l := values) (i := position)).symm
      simp [numbers, ended]
  | cons head tail ih =>
      let current := setLocal env "v" (number head)
      have captured : Binds current captures := binds_v env captures head bound noShadow
      have tested := checked collected current head captured rfl
      have nextSuffix : values.drop (position+1) = tail := by simp [List.drop_add_one_eq_tail_drop, suffix]
      have currentItem : (numbers values)[position]? = some (number head) := by
        have item : values[position]? = some head := by
          simpa [suffix] using (List.head?_drop (l := values) (i := position)).symm
        simp [numbers, item]
      cases keep : accept head with
      | true =>
          have appended : BlockContinues program current [.comprehensionAppend heap.length (.name "v")]
              (working heap collected) current (working heap (collected ++ [head])) globals := by
            have write := BlockContinues.comprehension_append
              (globals := globals) (module := program) (env := current) (address := heap.length)
              (items := numbers collected)
              (heap := working heap collected) (Evaluates.lookup (name := "v") (value := number head) rfl)
              (by simp [working])
            simpa [working, numbers, List.map_append, List.set_append_right] using write
          have emitted : BlockContinues program current (filterBody heap.length test)
              (working heap collected) current (working heap (collected ++ [head])) globals := by
            apply BlockContinues.branch (flag := true) (value := flag true) (middle := working heap collected)
            · simpa only [keep] using tested
            · rfl
            · exact appended
          obtain ⟨finalEnv, rest⟩ := ih (position+1) nextSuffix (collected ++ [head]) current captured
          refine ⟨finalEnv, ?_⟩
          apply BlockContinues.sequence_step (source := .ref address) (value := number head) (sequence collected) currentItem
          simpa [filterBody, Calls.loopContinuation, Calls.loopControl, List.filter_cons, keep, List.append_assoc] using
            emitted.append rest
      | false =>
          have emitted : BlockContinues program current (filterBody heap.length test)
              (working heap collected) current (working heap collected) globals := by
            apply BlockContinues.branch (flag := false) (value := flag false) (middle := working heap collected)
            · simpa only [keep] using tested
            · rfl
            · exact BlockContinues.nil
          obtain ⟨finalEnv, rest⟩ := ih (position+1) nextSuffix collected current captured
          refine ⟨finalEnv, ?_⟩
          apply BlockContinues.sequence_step (source := .ref address) (value := number head) (sequence collected) currentItem
          simpa [filterBody, Calls.loopContinuation, Calls.loopControl, List.filter_cons, keep] using emitted.append rest

/-- A real list comprehension allocates one private list and returns its stable filtered contents. -/
theorem filtered_list (heap : Heap) (env : HeapCalls.Env) (address : Nat) (values : List Int)
    (test : Expr) (accept : Int → Bool) (captures : List (String × Mutable.Value)) (globals : Mutable.Globals)
    (getValues : env "values" = some (.ref address)) (found : heap[address]? = some (.list (numbers values)))
    (noShadow : ∀ capture ∈ captures, capture.1 ≠ "v")
    (checked : ∀ (collected : List Int) (inner : HeapCalls.Env) (value : Int), Binds inner captures →
      inner "v" = some (number value) →
      Evaluates program inner test (working heap collected) (flag (accept value)) (working heap collected) globals)
    (bound : Binds env captures) :
    Evaluates program env (.listComp (.name "v") [("v", .name "values", [test])]) heap (.ref heap.length)
      (heap ++ [.list (numbers (values.filter accept))]) globals := by
  obtain ⟨finalEnv, executed⟩ := collect_filter heap address values values found 0 rfl []
    (comprehensionEnv env [("v", .name "values", [test])]) test accept captures globals noShadow checked
    (binds_comprehension env captures test bound noShadow)
  apply Evaluates.list_comp_sequence (ordinary := by intro expressions impossible; cases impossible)
    (middle := heap) (items := numbers values) (finalEnv := finalEnv)
    (Evaluates.lookup getValues) (by simp [Mutable.sequence, found])
  simpa [working, filterBody, comprehensionFilters, comprehensionBody] using executed

def maximum : List Int → Int
  | [] => 0
  | first :: rest => rest.foldl max first

private theorem unbox_numbers (values : List Int) :
    ((numbers values).mapM (fun item => match item with
      | .scalar value => pure value
      | .ref _ => Mutable.unsupported) : Mutable.Eval (List Constant)) = pure (values.map Constant.int) := by
  simp only [numbers, List.mapM_map]
  change (values.mapM (fun value => pure (Constant.int value)) : Mutable.Eval (List Constant)) = _
  exact List.mapM_pure

theorem primitive_maximum (heap : Heap) (address : Nat) (first : Int) (rest : List Int)
    (found : heap[address]? = some (.list (numbers (first :: rest)))) :
    primitive (.builtin .max) [.ref address] heap = some (.ok (number (maximum (first :: rest))), heap) := by
  have converted := unbox_numbers (first :: rest)
  have computed : ScalarSet.maximum ((first :: rest).map Constant.int) = .ok (.int (maximum (first :: rest))) := by
    simpa only [List.map_cons, ScalarSet.maximum, maximum] using ScalarSet.maximumFrom_ints rest first
  have execution (items : List Mutable.Value) (present : heap[address]? = some (.list items)) :
      primitive (.builtin .max) [.ref address] heap =
        ((do
          let scalars ← items.mapM (fun item => match item with
            | .scalar value => pure value
            | .ref _ => Mutable.unsupported)
          scalarResult (some (ScalarSet.maximum scalars))) : Mutable.Eval Mutable.Value) heap := by
    simp [primitive, Mutable.sequence, present]
    rfl
  rw [execution (numbers (first :: rest)) found, converted]
  change scalarResult (some (ScalarSet.maximum ((first :: rest).map Constant.int))) heap = _
  rw [computed]
  rfl

/-- A nonempty filtered source list is maximized without changing the caller heap or its collector. -/
theorem filtered_maximum (heap : Heap) (env : HeapCalls.Env) (address : Nat) (values : List Int)
    (test : Expr) (accept : Int → Bool) (captures : List (String × Mutable.Value)) (globals : Mutable.Globals)
    (getValues : env "values" = some (.ref address)) (found : heap[address]? = some (.list (numbers values)))
    (noShadow : ∀ capture ∈ captures, capture.1 ≠ "v")
    (checked : ∀ (collected : List Int) (inner : HeapCalls.Env) (value : Int), Binds inner captures →
      inner "v" = some (number value) →
      Evaluates program inner test (working heap collected) (flag (accept value)) (working heap collected) globals)
    (bound : Binds env captures) (nonempty : values.filter accept ≠ []) :
    Evaluates program env (.builtinCall .max [.listComp (.name "v") [("v", .name "values", [test])]]) heap
      (number (maximum (values.filter accept))) (heap ++ [.list (numbers (values.filter accept))]) globals := by
  cases equation : values.filter accept with
  | nil => exact False.elim (nonempty equation)
  | cons first rest =>
      apply Evaluates.builtin (.cons (filtered_list heap env address values test accept captures globals getValues found noShadow checked bound) .nil)
      have present : (heap ++ [Mutable.Object.list (numbers (first :: rest))])[heap.length]? =
          some (.list (numbers (first :: rest))) := by simp
      rw [equation]
      exact primitive_maximum (heap ++ [.list (numbers (first :: rest))]) heap.length first rest present

private abbrev fourTest : Expr := .compare (.name "v") [(.notEq, .name "four_val")]

private theorem four_test (heap : Heap) (inner : HeapCalls.Env) (value four : Int) (globals : Mutable.Globals)
    (getV : inner "v" = some (number value)) (getFour : inner "four_val" = some (number four)) :
    Evaluates program inner fourTest heap (flag (value != four)) heap globals := by
  apply Evaluates.compare (av := number value) (bv := number four) (middle := heap) (afterRight := heap)
    (Evaluates.lookup getV) (Evaluates.lookup getFour)
  by_cases same : value = four
  · subst four
    simp [HeapCalls.compare, scalarCompare, evalCmpOp, flag]
  · simp [HeapCalls.compare, scalarCompare, evalCmpOp, flag, same]

/-- This is the actual source quad branch, including the filtered comprehension, max, and return allocation. -/
theorem four_of_a_kind (heap : Heap) (env : HeapCalls.Env) (valuesAddress groupsAddress firstAddress : Nat)
    (values : List Int) (groups : List Mutable.Value) (four : Int) (globals : Mutable.Globals)
    (getValues : env "values" = some (.ref valuesAddress)) (getGroups : env "groups" = some (.ref groupsAddress))
    (valuesFound : heap[valuesAddress]? = some (.list (numbers values)))
    (groupsFound : heap[groupsAddress]? = some (.list groups)) (first : groups[0]? = some (.ref firstAddress))
    (fourFound : heap[firstAddress]? = some (.tuple [number 4, number four]))
    (nonempty : values.filter (fun value => value != four) ≠ []) :
    let filtered := heap ++ [Mutable.Object.list (numbers (values.filter (fun value => value != four)))]
    BlockReturns program env (code.drop 9) heap (.ref filtered.length)
      (filtered ++ [.tuple [number 7, number four, number (maximum (values.filter (fun value => value != four)))]]) globals := by
  let ranked := setLocal env "four_val" (number four)
  let captures : List (String × Mutable.Value) := [("four_val", number four)]
  have noShadow : ∀ capture ∈ captures, capture.1 ≠ "v" := by simp [captures]
  have captured : Binds ranked captures := by simp [Binds, captures, ranked, setLocal]
  have checked : ∀ (collected : List Int) (inner : HeapCalls.Env) (value : Int), Binds inner captures →
      inner "v" = some (number value) →
      Evaluates program inner fourTest (working heap collected) (flag (value != four)) (working heap collected) globals := by
    intro collected inner value bounds getV
    exact four_test (working heap collected) inner value four globals getV (bounds ("four_val", number four) (by simp [captures]))
  apply BlockReturns.branch (flag := true) (value := flag true) (middle := heap)
  · exact Evaluates.compare
      (straight_detection.group_count heap env groupsAddress 0 firstAddress groups 4 four globals getGroups groupsFound first fourFound)
      Evaluates.constant rfl
  · rfl
  · apply BlockReturns.assign
      (straight_detection.group_rank heap env groupsAddress 0 firstAddress groups 4 four globals getGroups groupsFound first fourFound)
    apply BlockReturns.assign
      (filtered_maximum heap ranked valuesAddress values fourTest (fun value => value != four) captures globals
        (by simpa [ranked, setLocal] using getValues) valuesFound noShadow checked captured nonempty)
    apply BlockReturns.return_value
    exact Evaluates.tuple (.cons Evaluates.constant
      (.cons (Evaluates.lookup (by simp [ranked, setLocal])) (.cons (Evaluates.lookup rfl) .nil))) (by simp)

theorem grouped_four_nonempty (heap : Heap) (values : List Int) (firstAddress : Nat) (four : Int)
    (five : values.length = 5)
    (first : (rank_groups.resultGroups (straight_detection.countedHeap heap values) (count_values.counts values))[0]? =
      some (.ref firstAddress))
    (fourFound : (straight_detection.groupedHeap heap values)[firstAddress]? = some (.tuple [number 4, number four])) :
    values.filter (fun value => value != four) ≠ [] := by
  obtain ⟨entry, entryMember, address, reference, found⟩ :=
    straight_detection.grouped_member heap values (.ref firstAddress) (List.mem_of_getElem? first)
  have sameAddress : firstAddress = address := by simpa using reference
  subst address
  have fields : entry.2 = 4 ∧ entry.1 = four := by
    simpa [number] using found.symm.trans fourFound
  have counted := Kickers.count_entry values entry entryMember
  rw [fields.1, fields.2] at counted
  exact Kickers.four_kicker_nonempty values four five (by omega)

/-- Actual source count groups discharge the kicker precondition, rather than assuming that max is safe. -/
theorem four_of_a_kind_grouped (heap : Heap) (env : HeapCalls.Env) (valuesAddress firstAddress : Nat)
    (values : List Int) (four : Int) (globals : Mutable.Globals) (five : values.length = 5)
    (getValues : env "values" = some (.ref valuesAddress))
    (getGroups : env "groups" = some (.ref
      (rank_groups.collected (straight_detection.countedHeap heap values) (count_values.counts values)).length))
    (valuesFound : (straight_detection.groupedHeap heap values)[valuesAddress]? = some (.list (numbers values)))
    (first : (rank_groups.resultGroups (straight_detection.countedHeap heap values) (count_values.counts values))[0]? =
      some (.ref firstAddress))
    (fourFound : (straight_detection.groupedHeap heap values)[firstAddress]? = some (.tuple [number 4, number four])) :
    let filtered := straight_detection.groupedHeap heap values ++ [.list (numbers (values.filter (fun value => value != four)))]
    BlockReturns program env (code.drop 9) (straight_detection.groupedHeap heap values) (.ref filtered.length)
      (filtered ++ [.tuple [number 7, number four, number (maximum (values.filter (fun value => value != four)))]]) globals := by
  exact four_of_a_kind (straight_detection.groupedHeap heap values) env valuesAddress
    (rank_groups.collected (straight_detection.countedHeap heap values) (count_values.counts values)).length
    firstAddress values (rank_groups.resultGroups (straight_detection.countedHeap heap values) (count_values.counts values)) four
    globals getValues getGroups valuesFound (straight_detection.grouped_found heap values) first fourFound
    (grouped_four_nonempty heap values firstAddress four five first fourFound)

end solvers.«54».numeric_score.repeated_categories.quad_path

namespace solvers.«54».numeric_score.priority_categories
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
set_option Elab.async false

private abbrev program := Solvers.P54Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
private abbrev numbers (values : List Int) : List Mutable.Value := values.map number
private abbrev flag (value : Bool) : Mutable.Value := .scalar (.bool value)
private def code : List Stmt := executableBody ((findFunctionIn program.body "_score_values").getD ([], [])).2

private theorem equal_integer (heap : Heap) (env : HeapCalls.Env) (expression : Expr) (value wanted : Int)
    (globals : Mutable.Globals) (evaluated : Evaluates program env expression heap (number value) heap globals) :
    Evaluates program env (.compare expression [(.eq, .constant (.int wanted))])
      heap (flag (decide (value = wanted))) heap globals := by
  apply Evaluates.compare (av := number value) (bv := number wanted) (middle := heap) (afterRight := heap)
    evaluated Evaluates.constant
  simp [HeapCalls.compare, scalarCompare, evalCmpOp, flag]

private theorem values_tuple (heap : Heap) (env : HeapCalls.Env) (address : Nat) (values : List Int)
    (globals : Mutable.Globals) (nonempty : values ≠ [])
    (bound : env "values" = some (.ref address)) (found : heap[address]? = some (.list (numbers values))) :
    Evaluates program env (.builtinCall .tuple [.name "values"]) heap (.ref heap.length)
      (heap ++ [.tuple (numbers values)]) globals := by
  apply Evaluates.builtin (.cons (Evaluates.lookup bound) .nil)
  have converted := primitive_tuple_list heap address (numbers values) found
  cases values with
  | nil => cases nonempty rfl
  | cons first rest =>
      simpa [Mutable.allocateTuple, numbers, Except.map] using converted

private theorem flush_rank (heap : Heap) (env : HeapCalls.Env) (address : Nat) (values : List Int)
    (globals : Mutable.Globals) (nonempty : values ≠ [])
    (bound : env "values" = some (.ref address)) (found : heap[address]? = some (.list (numbers values))) :
    Evaluates program env (.tuple [.constant (.int 5), .builtinCall .tuple [.name "values"]]) heap
      (.ref (heap.length+1))
      (heap ++ [.tuple (numbers values), .tuple [number 5, .ref heap.length]]) globals := by
  have constructed : Evaluates program env (.tuple [.constant (.int 5), .builtinCall .tuple [.name "values"]]) heap
      (.ref (heap ++ [Mutable.Object.tuple (numbers values)]).length)
      ((heap ++ [.tuple (numbers values)]) ++ [.tuple [number 5, .ref heap.length]]) globals :=
    Evaluates.tuple (.cons Evaluates.constant (.cons (values_tuple heap env address values globals nonempty bound found) .nil))
      (by simp)
  simpa only [List.length_append, List.length_singleton, List.append_assoc] using constructed

/-- The real priority suffix returns before any triple or pair branch, for arbitrary group counts. -/
theorem result (heap : Heap) (env : HeapCalls.Env) (valuesAddress : Nat)
    (values : List Int) (high : Int) (flush straight : Bool) (globals : Mutable.Globals)
    (nonempty : values ≠ [])
    (getValues : env "values" = some (.ref valuesAddress))
    (valuesFound : heap[valuesAddress]? = some (.list (numbers values)))
    (getFlush : env "is_flush" = some (flag flush))
    (getStraight : env "is_straight" = some (flag straight))
    (getHigh : env "straight_high" = some (number high))
    (priority : flush || straight = true) :
    BlockReturns program env (code.drop 11) heap (simple_categories.answer heap flush straight)
      (simple_categories.output heap values flush straight high) globals := by
  cases flush with
  | true =>
      simp only [simple_categories.answer, simple_categories.output, ↓reduceIte]
      apply BlockReturns.branch (flag := true) (value := flag true) (middle := heap) (Evaluates.lookup getFlush) rfl
      exact BlockReturns.return_value (flush_rank heap env valuesAddress values globals nonempty getValues valuesFound)
  | false =>
      have straightTrue : straight = true := by simpa using priority
      subst straight
      simp only [simple_categories.answer, simple_categories.output, ↓reduceIte]
      apply BlockReturns.branch (flag := false) (value := flag false) (middle := heap) (Evaluates.lookup getFlush) rfl
      apply BlockReturns.branch (flag := true) (value := flag true) (middle := heap) (Evaluates.lookup getStraight) rfl
      apply BlockReturns.return_value
      exact Evaluates.tuple (.cons Evaluates.constant (.cons (Evaluates.lookup getHigh) .nil)) (by simp)

/-- A first count other than three or four skips both earlier categories without reading groups[1]. -/
theorem without_triple_or_quad (heap : Heap) (env : HeapCalls.Env)
    (valuesAddress groupsAddress firstAddress : Nat) (values : List Int) (groups : List Mutable.Value)
    (count rank high : Int) (flush straight : Bool) (globals : Mutable.Globals)
    (nonempty : values ≠ [])
    (getValues : env "values" = some (.ref valuesAddress))
    (valuesFound : heap[valuesAddress]? = some (.list (numbers values)))
    (getGroups : env "groups" = some (.ref groupsAddress))
    (groupsFound : heap[groupsAddress]? = some (.list groups))
    (first : groups[0]? = some (.ref firstAddress))
    (pair : heap[firstAddress]? = some (.tuple [number count, number rank]))
    (getFlush : env "is_flush" = some (flag flush))
    (getStraight : env "is_straight" = some (flag straight))
    (getHigh : env "straight_high" = some (number high))
    (notThree : count ≠ 3) (notFour : count ≠ 4) (priority : flush || straight = true) :
    BlockReturns program env (code.drop 9) heap (simple_categories.answer heap flush straight)
      (simple_categories.output heap values flush straight high) globals := by
  have counted := straight_detection.group_count heap env groupsAddress 0 firstAddress groups count rank globals
    getGroups groupsFound first pair
  have noFour : Evaluates program env
      (.compare (straight_detection.groupExpression 0 0) [(.eq, .constant (.int 4))]) heap (flag false) heap globals := by
    simpa only [notFour, decide_false] using
      equal_integer heap env (straight_detection.groupExpression 0 0) count 4 globals counted
  have noThree : Evaluates program env
      (.compare (straight_detection.groupExpression 0 0) [(.eq, .constant (.int 3))]) heap (flag false) heap globals := by
    simpa only [notThree, decide_false] using
      equal_integer heap env (straight_detection.groupExpression 0 0) count 3 globals counted
  apply BlockReturns.branch (flag := false) (value := flag false) (middle := heap) noFour rfl
  apply BlockReturns.branch (flag := false) (value := flag false) (middle := heap)
    (Evaluates.boolean_stop (flag := false) noThree rfl rfl) rfl
  exact result heap env valuesAddress values high flush straight globals nonempty getValues valuesFound
    getFlush getStraight getHigh priority

/-- A triple with a non-pair second group reaches flush/straight priority after the real full-house test. -/
theorem triple_without_full_house (heap : Heap) (env : HeapCalls.Env)
    (valuesAddress groupsAddress firstAddress secondAddress : Nat) (values : List Int) (groups : List Mutable.Value)
    (trip secondCount secondRank high : Int) (flush straight : Bool) (globals : Mutable.Globals)
    (nonempty : values ≠ [])
    (getValues : env "values" = some (.ref valuesAddress))
    (valuesFound : heap[valuesAddress]? = some (.list (numbers values)))
    (getGroups : env "groups" = some (.ref groupsAddress))
    (groupsFound : heap[groupsAddress]? = some (.list groups))
    (first : groups[0]? = some (.ref firstAddress))
    (second : groups[1]? = some (.ref secondAddress))
    (tripFound : heap[firstAddress]? = some (.tuple [number 3, number trip]))
    (secondFound : heap[secondAddress]? = some (.tuple [number secondCount, number secondRank]))
    (getFlush : env "is_flush" = some (flag flush))
    (getStraight : env "is_straight" = some (flag straight))
    (getHigh : env "straight_high" = some (number high))
    (notFullHouse : secondCount ≠ 2) (priority : flush || straight = true) :
    BlockReturns program env (code.drop 9) heap (simple_categories.answer heap flush straight)
      (simple_categories.output heap values flush straight high) globals := by
  have firstCount := straight_detection.group_count heap env groupsAddress 0 firstAddress groups 3 trip globals
    getGroups groupsFound first tripFound
  have secondCounted := straight_detection.group_count heap env groupsAddress 1 secondAddress groups secondCount secondRank globals
    getGroups groupsFound second secondFound
  have noPair : Evaluates program env
      (.compare (straight_detection.groupExpression 1 0) [(.eq, .constant (.int 2))]) heap (flag false) heap globals := by
    simpa only [notFullHouse, decide_false] using
      equal_integer heap env (straight_detection.groupExpression 1 0) secondCount 2 globals secondCounted
  apply BlockReturns.branch (flag := false) (value := flag false) (middle := heap)
    (Evaluates.compare firstCount Evaluates.constant rfl) rfl
  apply BlockReturns.branch (flag := false) (value := flag false) (middle := heap)
    (Evaluates.boolean_next (flag := true)
      (Evaluates.compare firstCount Evaluates.constant rfl) rfl rfl
      (Evaluates.boolean_last (flag := false) noPair rfl)) rfl
  exact result heap env valuesAddress values high flush straight globals nonempty getValues valuesFound
    getFlush getStraight getHigh priority

end solvers.«54».numeric_score.priority_categories

namespace solvers.«54».numeric_score.two_pairs
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
set_option Elab.async false

private abbrev program := Solvers.P54Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
private abbrev numbers (values : List Int) : List Mutable.Value := values.map number
private abbrev flag (value : Bool) : Mutable.Value := .scalar (.bool value)
private def code : List Stmt := executableBody ((findFunctionIn program.body "_score_values").getD ([], [])).2
private abbrev firstRank := straight_detection.groupExpression 0 1
private abbrev secondRank := straight_detection.groupExpression 1 1
private abbrev filterTest : Expr := .boolOp .and
  [.compare (.name "v") [(.notEq, .name "pair_high")],
   .compare (.name "v") [(.notEq, .name "pair_low")]]

private theorem maximum_two (heap : Heap) (left right : Int) :
    primitive (.builtin .max) [number left, number right] heap =
      some (.ok (number (max left right)), heap) := by
  by_cases larger : left < right
  · simp [primitive, Calls.positionalMaximum, Calls.maximumFrom, scalarCompare, evalCmpOp,
      scalarResult, number, larger, Int.max_eq_right (Int.le_of_lt larger)]
  · simp [primitive, Calls.positionalMaximum, Calls.maximumFrom, scalarCompare, evalCmpOp,
      scalarResult, number, larger, Int.max_eq_left (Int.le_of_not_gt larger)]

private theorem minimum_two (heap : Heap) (left right : Int) :
    primitive (.builtin .min) [number left, number right] heap =
      some (.ok (number (min left right)), heap) := by
  by_cases smaller : right < left
  · simp [primitive, Calls.positionalMinimum, Calls.minimumFrom, scalarCompare, evalCmpOp,
      scalarResult, number, smaller, Int.min_eq_right (Int.le_of_lt smaller)]
  · simp [primitive, Calls.positionalMinimum, Calls.minimumFrom, scalarCompare, evalCmpOp,
      scalarResult, number, smaller, Int.min_eq_left (Int.le_of_not_gt smaller)]

private theorem filter_evaluates (heap : Heap) (env : HeapCalls.Env) (value high low : Int)
    (globals : Mutable.Globals) (getV : env "v" = some (number value))
    (getHigh : env "pair_high" = some (number high)) (getLow : env "pair_low" = some (number low)) :
    Evaluates program env filterTest heap (flag (value != high && value != low)) heap globals := by
  have first : Evaluates program env (.compare (.name "v") [(.notEq, .name "pair_high")])
      heap (flag (value != high)) heap globals := by
    apply Evaluates.compare (av := number value) (bv := number high) (middle := heap) (afterRight := heap)
      (Evaluates.lookup getV) (Evaluates.lookup getHigh)
    by_cases same : value = high <;> simp [HeapCalls.compare, scalarCompare, evalCmpOp, flag, same]
  have second : Evaluates program env (.compare (.name "v") [(.notEq, .name "pair_low")])
      heap (flag (value != low)) heap globals := by
    apply Evaluates.compare (av := number value) (bv := number low) (middle := heap) (afterRight := heap)
      (Evaluates.lookup getV) (Evaluates.lookup getLow)
    by_cases same : value = low <;> simp [HeapCalls.compare, scalarCompare, evalCmpOp, flag, same]
  cases accepted : value != high with
  | false =>
      simp only [Bool.false_and]
      exact Evaluates.boolean_stop (flag := false) (by simpa only [accepted] using first) rfl rfl
  | true =>
      simp only [Bool.true_and]
      exact Evaluates.boolean_next (flag := true) (by simpa only [accepted] using first) rfl rfl
        (Evaluates.boolean_last (flag := value != low) second rfl)

def remaining (values : List Int) (left right : Int) : List Int :=
  values.filter fun value => value != max left right && value != min left right

theorem remaining_eq (values : List Int) (left right : Int) :
    remaining values left right = values.filter (fun value => value != left && value != right) := by
  unfold remaining
  by_cases ordered : left ≤ right
  · simp only [Int.max_eq_right ordered, Int.min_eq_left ordered]
    congr 1
    funext value
    exact Bool.and_comm _ _
  · have reverse : right ≤ left := by omega
    rw [Int.max_eq_left reverse, Int.min_eq_right reverse]

/-- The actual two-pair path computes both rank extrema and the nonempty filtered kicker. -/
theorem result (heap : Heap) (env : HeapCalls.Env) (valuesAddress groupsAddress firstAddress secondAddress : Nat)
    (values : List Int) (groups : List Mutable.Value) (left right : Int) (globals : Mutable.Globals)
    (getValues : env "values" = some (.ref valuesAddress))
    (valuesFound : heap[valuesAddress]? = some (.list (numbers values)))
    (getGroups : env "groups" = some (.ref groupsAddress))
    (groupsFound : heap[groupsAddress]? = some (.list groups))
    (first : groups[0]? = some (.ref firstAddress)) (second : groups[1]? = some (.ref secondAddress))
    (leftFound : heap[firstAddress]? = some (.tuple [number 2, number left]))
    (rightFound : heap[secondAddress]? = some (.tuple [number 2, number right]))
    (getFlush : env "is_flush" = some (flag false)) (getStraight : env "is_straight" = some (flag false))
    (nonempty : remaining values left right ≠ []) :
    BlockReturns program env (code.drop 9) heap (.ref (heap.length+1))
      (heap ++ [.list (numbers (remaining values left right)),
        .tuple [number 2, number (max left right), number (min left right),
          number (repeated_categories.quad_path.maximum (remaining values left right))]]) globals := by
  have firstCount := straight_detection.group_count heap env groupsAddress 0 firstAddress groups 2 left globals
    getGroups groupsFound first leftFound
  have secondCount := straight_detection.group_count heap env groupsAddress 1 secondAddress groups 2 right globals
    getGroups groupsFound second rightFound
  let highEnv := setLocal env "pair_high" (number (max left right))
  let ready := setLocal highEnv "pair_low" (number (min left right))
  have highComputed : Evaluates program env (.builtinCall .max [firstRank, secondRank])
      heap (number (max left right)) heap globals := by
    apply Evaluates.builtin
      (.cons (straight_detection.group_rank heap env groupsAddress 0 firstAddress groups 2 left globals
        getGroups groupsFound first leftFound)
        (.cons (straight_detection.group_rank heap env groupsAddress 1 secondAddress groups 2 right globals
          getGroups groupsFound second rightFound) .nil))
    exact maximum_two heap left right
  have lowComputed : Evaluates program highEnv (.builtinCall .min [firstRank, secondRank])
      heap (number (min left right)) heap globals := by
    have bound : highEnv "groups" = some (.ref groupsAddress) := by simpa [highEnv, setLocal] using getGroups
    apply Evaluates.builtin
      (.cons (straight_detection.group_rank heap highEnv groupsAddress 0 firstAddress groups 2 left globals
        bound groupsFound first leftFound)
        (.cons (straight_detection.group_rank heap highEnv groupsAddress 1 secondAddress groups 2 right globals
          bound groupsFound second rightFound) .nil))
    exact minimum_two heap left right
  have kicker := repeated_categories.quad_path.filtered_maximum heap ready valuesAddress values filterTest
    (fun value => value != max left right && value != min left right)
    [("pair_high", number (max left right)), ("pair_low", number (min left right))] globals
    (by simpa [ready, highEnv, setLocal] using getValues) valuesFound
    (by simp)
    (by
      intro collected inner value captures getV
      exact filter_evaluates _ inner value (max left right) (min left right) globals getV
        (captures ("pair_high", number (max left right)) (by simp))
        (captures ("pair_low", number (min left right)) (by simp)))
    (by simp [repeated_categories.quad_path.Binds, ready, highEnv, setLocal]) nonempty
  apply BlockReturns.branch (flag := false) (value := flag false) (middle := heap)
    (Evaluates.compare firstCount Evaluates.constant rfl) rfl
  apply BlockReturns.branch (flag := false) (value := flag false) (middle := heap)
    (Evaluates.boolean_stop (flag := false) (Evaluates.compare firstCount Evaluates.constant rfl) rfl rfl) rfl
  apply BlockReturns.branch (flag := false) (value := flag false) (middle := heap) (Evaluates.lookup getFlush) rfl
  apply BlockReturns.branch (flag := false) (value := flag false) (middle := heap) (Evaluates.lookup getStraight) rfl
  apply BlockReturns.branch (flag := false) (value := flag false) (middle := heap)
    (Evaluates.compare firstCount Evaluates.constant rfl) rfl
  apply BlockReturns.branch (flag := true) (value := flag true) (middle := heap)
    (Evaluates.boolean_next (flag := true) (Evaluates.compare firstCount Evaluates.constant rfl) rfl rfl
      (Evaluates.boolean_last (flag := true) (Evaluates.compare secondCount Evaluates.constant rfl) rfl)) rfl
  apply BlockReturns.assign highComputed
  apply BlockReturns.assign lowComputed
  apply BlockReturns.assign kicker
  apply BlockReturns.return_value
  have built := Evaluates.tuple (module := program) (globals := globals)
    (env := setLocal ready "kicker" (number (repeated_categories.quad_path.maximum (remaining values left right))))
    (heap := heap ++ [.list (numbers (remaining values left right))])
    (.cons (Evaluates.constant (value := .int 2))
      (.cons (Evaluates.lookup (name := "pair_high") (value := number (max left right)) (by simp [ready, highEnv, setLocal]))
        (.cons (Evaluates.lookup (name := "pair_low") (value := number (min left right)) (by simp [ready, setLocal]))
          (.cons (Evaluates.lookup (name := "kicker")
            (value := number (repeated_categories.quad_path.maximum (remaining values left right))) rfl) .nil)))) (by simp)
  simpa only [List.length_append, List.length_singleton, List.append_assoc] using built

end solvers.«54».numeric_score.two_pairs

namespace solvers.«54».numeric_score.totality
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
set_option Elab.async false

private abbrev program := Solvers.P54Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
private abbrev numbers (values : List Int) : List Mutable.Value := values.map number
private abbrev flag (value : Bool) : Mutable.Value := .scalar (.bool value)
private def code : List Stmt := executableBody ((findFunctionIn program.body "_score_values").getD ([], [])).2

def groups (heap : Heap) (values : List Int) : List Mutable.Value :=
  rank_groups.resultGroups (straight_detection.countedHeap heap values) (count_values.counts values)

def groupsAddress (heap : Heap) (values : List Int) : Nat :=
  (rank_groups.collected (straight_detection.countedHeap heap values) (count_values.counts values)).length

def frequencies (heap : Heap) (values : List Int) : List Int :=
  (groups heap values).map fun group =>
    (ProjectEulerSolutions.TupleSorting.readInts (straight_detection.groupedHeap heap values) group).headD 0

/-- A successful frequency lookup decodes the actual source tuple and its exact input count. -/
theorem group_at (heap : Heap) (values : List Int) (index : Nat) (count : Int)
    (atCount : (frequencies heap values)[index]? = some count) :
    ∃ address rank,
      (groups heap values)[index]? = some (.ref address) ∧
      (straight_detection.groupedHeap heap values)[address]? = some (.tuple [number count, number rank]) ∧
      (values.count rank : Int) = count := by
  change ((groups heap values).map _)[index]? = some count at atCount
  rw [List.getElem?_map] at atCount
  cases atGroup : (groups heap values)[index]? with
  | none => simp [atGroup] at atCount
  | some group =>
      have frequency :
          (ProjectEulerSolutions.TupleSorting.readInts (straight_detection.groupedHeap heap values) group).headD 0 = count := by
        simpa only [atGroup, Option.map_some, Option.some.injEq] using atCount
      obtain ⟨entry, member, address, reference, found⟩ :=
        straight_detection.grouped_member heap values group (List.mem_of_getElem? atGroup)
      subst group
      have field : entry.2 = count := by
        simpa [ProjectEulerSolutions.TupleSorting.readInts, found, number] using frequency
      have counted := Kickers.count_entry values entry member
      refine ⟨address, entry.1, rfl, ?_, ?_⟩
      · simpa only [field] using found
      · exact counted.symm.trans field

theorem grouped_preserves (heap : Heap) (values : List Int) (address : Nat) (object : Mutable.Object)
    (found : heap[address]? = some object) :
    (straight_detection.groupedHeap heap values)[address]? = some object := by
  unfold straight_detection.groupedHeap
  apply ProjectEulerSolutions.HeapCollection.prefix_found
  change ((straight_detection.countedHeap heap values) ++
    (.list (rank_groups.references ((straight_detection.countedHeap heap values).length + 1)
      (count_values.counts values)) :: rank_groups.scratch (count_values.counts values)))[address]? = some object
  apply ProjectEulerSolutions.HeapCollection.prefix_found
  unfold straight_detection.countedHeap
  exact ProjectEulerSolutions.HeapCollection.prefix_found _ _ address object found

/-- All seven actual five-value profiles select a proved executable category path. -/
theorem suffix_total (heap : Heap) (env : HeapCalls.Env) (valuesAddress : Nat)
    (values : List Int) (high : Int) (flush straight : Bool) (globals : Mutable.Globals)
    (five : values.length = 5)
    (getValues : env "values" = some (.ref valuesAddress))
    (valuesFound : (straight_detection.groupedHeap heap values)[valuesAddress]? = some (.list (numbers values)))
    (getGroups : env "groups" = some (.ref (groupsAddress heap values)))
    (getFlush : env "is_flush" = some (flag flush))
    (getStraight : env "is_straight" = some (flag straight))
    (getHigh : env "straight_high" = some (number high)) :
    ∃ answer finalHeap, BlockReturns program env (code.drop 9)
      (straight_detection.groupedHeap heap values) answer finalHeap globals := by
  have nonempty : values ≠ [] := by intro empty; subst values; simp at five
  have listFound := straight_detection.grouped_found heap values
  have profile := straight_detection.grouped_profile heap values five
  simp only [List.map_map] at profile
  change frequencies heap values ∈ ProjectEulerSolutions.PokerCountPatterns.profiles at profile
  simp only [ProjectEulerSolutions.PokerCountPatterns.profiles, List.mem_cons, List.not_mem_nil, or_false] at profile
  rcases profile with allEqual | quad | fullHouse | triple | twoPairs | onePair | singles
  · obtain ⟨address, rank, first, pair, counted⟩ := group_at heap values 0 5 (by rw [allEqual]; rfl)
    exact ⟨_, _, simple_categories.result (straight_detection.groupedHeap heap values) env
      valuesAddress (groupsAddress heap values) address values (groups heap values) 5 rank high flush straight globals
      nonempty getValues valuesFound getGroups listFound first pair getFlush getStraight getHigh
      (by decide) (by decide) (by decide)⟩
  · obtain ⟨address, rank, first, pair, counted⟩ := group_at heap values 0 4 (by rw [quad]; rfl)
    have remains : values.filter (fun value => value != rank) ≠ [] :=
      Kickers.four_kicker_nonempty values rank five (by omega)
    exact ⟨_, _, repeated_categories.quad_path.four_of_a_kind (straight_detection.groupedHeap heap values) env
      valuesAddress (groupsAddress heap values) address values (groups heap values) rank globals
      getValues getGroups valuesFound listFound first pair remains⟩
  · obtain ⟨firstAddress, trip, first, tripFound, tripCount⟩ := group_at heap values 0 3 (by rw [fullHouse]; rfl)
    obtain ⟨secondAddress, pair, second, pairFound, pairCount⟩ := group_at heap values 1 2 (by rw [fullHouse]; rfl)
    exact ⟨_, _, repeated_categories.full_house (straight_detection.groupedHeap heap values) env
      (groupsAddress heap values) firstAddress secondAddress (groups heap values) trip pair globals
      getGroups listFound first second tripFound pairFound⟩
  · obtain ⟨firstAddress, rank, first, firstFound, firstCount⟩ := group_at heap values 0 3 (by rw [triple]; rfl)
    obtain ⟨secondAddress, secondRank, second, secondFound, secondCount⟩ := group_at heap values 1 1 (by rw [triple]; rfl)
    by_cases priority : flush || straight = true
    · exact ⟨_, _, priority_categories.triple_without_full_house (straight_detection.groupedHeap heap values) env
        valuesAddress (groupsAddress heap values) firstAddress secondAddress values (groups heap values)
        rank 1 secondRank high flush straight globals nonempty getValues valuesFound getGroups listFound
        first second firstFound secondFound getFlush getStraight getHigh (by decide) priority⟩
    · have noFlush : flush = false := by cases flush <;> cases straight <;> simp_all
      have noStraight : straight = false := by cases flush <;> cases straight <;> simp_all
      have remains : filtered_categories.filtered values rank ≠ [] :=
        filtered_categories.nonempty_three values rank five (by omega)
      exact ⟨_, _, filtered_categories.trips (straight_detection.groupedHeap heap values) env
        valuesAddress (groupsAddress heap values) firstAddress secondAddress values (groups heap values) rank 1 secondRank globals
        getValues valuesFound getGroups listFound first second firstFound secondFound (by decide)
        (by simpa only [noFlush] using getFlush) (by simpa only [noStraight] using getStraight) remains⟩
  · obtain ⟨firstAddress, left, first, firstFound, firstCount⟩ := group_at heap values 0 2 (by rw [twoPairs]; rfl)
    obtain ⟨secondAddress, right, second, secondFound, secondCount⟩ := group_at heap values 1 2 (by rw [twoPairs]; rfl)
    by_cases priority : flush || straight = true
    · exact ⟨_, _, priority_categories.without_triple_or_quad (straight_detection.groupedHeap heap values) env
        valuesAddress (groupsAddress heap values) firstAddress values (groups heap values) 2 left high flush straight globals
        nonempty getValues valuesFound getGroups listFound first firstFound getFlush getStraight getHigh
        (by decide) (by decide) priority⟩
    · have noFlush : flush = false := by cases flush <;> cases straight <;> simp_all
      have noStraight : straight = false := by cases flush <;> cases straight <;> simp_all
      have remains : two_pairs.remaining values left right ≠ [] := by
        rw [two_pairs.remaining_eq]
        apply Kickers.two_pair_positions_kicker_nonempty (straight_detection.countedHeap heap values) heap.length values
          (by simp [straight_detection.countedHeap]) five 0 1 (.ref firstAddress) (.ref secondAddress)
          first second (by decide) left right
        · change ProjectEulerSolutions.TupleSorting.readInts (straight_detection.groupedHeap heap values) (.ref firstAddress) = [2, left]
          simp [ProjectEulerSolutions.TupleSorting.readInts, firstFound, number]
        · change ProjectEulerSolutions.TupleSorting.readInts (straight_detection.groupedHeap heap values) (.ref secondAddress) = [2, right]
          simp [ProjectEulerSolutions.TupleSorting.readInts, secondFound, number]
      exact ⟨_, _, two_pairs.result (straight_detection.groupedHeap heap values) env
        valuesAddress (groupsAddress heap values) firstAddress secondAddress values (groups heap values) left right globals
        getValues valuesFound getGroups listFound first second firstFound secondFound
        (by simpa only [noFlush] using getFlush) (by simpa only [noStraight] using getStraight) remains⟩
  · obtain ⟨firstAddress, rank, first, firstFound, firstCount⟩ := group_at heap values 0 2 (by rw [onePair]; rfl)
    obtain ⟨secondAddress, secondRank, second, secondFound, secondCount⟩ := group_at heap values 1 1 (by rw [onePair]; rfl)
    by_cases priority : flush || straight = true
    · exact ⟨_, _, priority_categories.without_triple_or_quad (straight_detection.groupedHeap heap values) env
        valuesAddress (groupsAddress heap values) firstAddress values (groups heap values) 2 rank high flush straight globals
        nonempty getValues valuesFound getGroups listFound first firstFound getFlush getStraight getHigh
        (by decide) (by decide) priority⟩
    · have noFlush : flush = false := by cases flush <;> cases straight <;> simp_all
      have noStraight : straight = false := by cases flush <;> cases straight <;> simp_all
      have remains : filtered_categories.filtered values rank ≠ [] :=
        filtered_categories.nonempty_pair values rank five (by omega)
      exact ⟨_, _, filtered_categories.pair (straight_detection.groupedHeap heap values) env
        valuesAddress (groupsAddress heap values) firstAddress secondAddress values (groups heap values) rank 1 secondRank globals
        getValues valuesFound getGroups listFound first second firstFound secondFound (by decide)
        (by simpa only [noFlush] using getFlush) (by simpa only [noStraight] using getStraight) remains⟩
  · obtain ⟨address, rank, first, pair, counted⟩ := group_at heap values 0 1 (by rw [singles]; rfl)
    exact ⟨_, _, simple_categories.result (straight_detection.groupedHeap heap values) env
      valuesAddress (groupsAddress heap values) address values (groups heap values) 1 rank high flush straight globals
      nonempty getValues valuesFound getGroups listFound first pair getFlush getStraight getHigh
      (by decide) (by decide) (by decide)⟩

/-- Every five-integer input and arbitrary string suit list terminates through the actual source helper. -/
theorem result (heap : Heap) (valuesAddress suitsAddress : Nat) (values : List Int) (suits : List String)
    (globals : Mutable.Globals) (five : values.length = 5)
    (valuesFound : heap[valuesAddress]? = some (.list (numbers values)))
    (suitsFound : heap[suitsAddress]? = some (.list (suits.map fun value => .scalar (.string value)))) :
    ∃ answer finalHeap, ReturnsValue program "_score_values" [.ref valuesAddress, .ref suitsAddress]
      heap answer finalHeap globals := by
  have nonempty : values ≠ [] := by intro empty; subst values; simp at five
  by_cases both : (straight_detection.detected (orderedUnique values) && isFlush suits) = true
  · obtain ⟨straight, flush⟩ := Bool.and_eq_true_iff.mp both
    exact ⟨_, _, straight_detection.straight_flush_result heap valuesAddress suitsAddress values suits globals
      nonempty valuesFound suitsFound straight flush⟩
  · have notBoth : (straight_detection.detected (orderedUnique values) && isFlush suits) = false := Bool.eq_false_iff.mpr both
    let start := composition.entry globals valuesAddress suitsAddress
    let prepared := composition.preparedHeap heap valuesAddress values suits
    let ready := composition.scoredEnv start heap valuesAddress values suits
    have getValues : ready "values" = some (.ref valuesAddress) := by
      change straight_detection.groupedEnv (composition.preparedEnv start heap values suits) prepared
        (descending values) "values" = _
      rw [straight_detection.grouped_other _ _ _ "values" (by decide) (by decide),
        composition.prepared_other _ _ _ _ "values" (by decide) (by decide) (by decide) (by decide)]
      cases globals <;> rfl
    have getGroups : ready "groups" = some (.ref (groupsAddress prepared (descending values))) := by
      simp [ready, composition.scoredEnv, straight_detection.groupedEnv, groupsAddress, prepared, setLocal]
    have getFlush : ready "is_flush" = some (flag (isFlush suits)) := by
      change straight_detection.groupedEnv (composition.preparedEnv start heap values suits) prepared
        (descending values) "is_flush" = _
      rw [straight_detection.grouped_other _ _ _ "is_flush" (by decide) (by decide)]
      exact composition.prepared_flush start heap values suits
    have getStraight : ready "is_straight" = some (flag (straight_detection.detected (orderedUnique values))) := by
      change straight_detection.groupedEnv (composition.preparedEnv start heap values suits) prepared
        (descending values) "is_straight" = _
      rw [straight_detection.grouped_other _ _ _ "is_straight" (by decide) (by decide)]
      exact composition.prepared_straight start heap values suits
    have getHigh : ready "straight_high" = some (number (straight_detection.high (orderedUnique values))) := by
      change straight_detection.groupedEnv (composition.preparedEnv start heap values suits) prepared
        (descending values) "straight_high" = _
      rw [straight_detection.grouped_other _ _ _ "straight_high" (by decide) (by decide)]
      exact straight_detection.updated_high (normalizationEnv start heap suits) (orderedUnique values)
        (by simp [normalizationEnv, setLocal])
    have sortedFive : (descending values).length = 5 :=
      (ProjectEulerSolutions.PokerUnique.descending_perm values).length_eq.trans five
    have originalValues : prepared[valuesAddress]? = some (.list (numbers (descending values))) :=
      composition.prepared_values heap valuesAddress values suits valuesFound
    obtain ⟨answer, finalHeap, suffix⟩ := suffix_total prepared ready valuesAddress (descending values)
      (straight_detection.high (orderedUnique values)) (isFlush suits) (straight_detection.detected (orderedUnique values))
      globals sortedFive getValues (grouped_preserves prepared (descending values) valuesAddress _ originalValues)
      getGroups getFlush getStraight getHigh
    exact ⟨answer, finalHeap, composition.result_of_suffix heap valuesAddress suitsAddress values suits globals
      nonempty valuesFound suitsFound notBoth answer finalHeap suffix⟩

theorem returns (heap : Heap) (valuesAddress suitsAddress : Nat) (values : List Int) (suits : List String)
    (globals : Mutable.Globals) (five : values.length = 5)
    (valuesFound : heap[valuesAddress]? = some (.list (numbers values)))
    (suitsFound : heap[suitsAddress]? = some (.list (suits.map fun value => .scalar (.string value)))) :
    (HeapCalls.call program "_score_values" [.ref valuesAddress, .ref suitsAddress] heap globals).Returns := by
  obtain ⟨answer, finalHeap, executed⟩ := result heap valuesAddress suitsAddress values suits globals five valuesFound suitsFound
  exact executed.returns

theorem supported (heap : Heap) (valuesAddress suitsAddress : Nat) (values : List Int) (suits : List String)
    (globals : Mutable.Globals) (five : values.length = 5)
    (valuesFound : heap[valuesAddress]? = some (.list (numbers values)))
    (suitsFound : heap[suitsAddress]? = some (.list (suits.map fun value => .scalar (.string value)))) :
    (HeapCalls.call program "_score_values" [.ref valuesAddress, .ref suitsAddress] heap globals).Supported :=
  (returns heap valuesAddress suitsAddress values suits globals five valuesFound suitsFound).supported

end solvers.«54».numeric_score.totality

namespace solvers.«54»._score_values
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
set_option Elab.async false

private abbrev program := Solvers.P54Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
private abbrev numbers (values : List Int) : List Mutable.Value := values.map number
private abbrev flag (value : Bool) : Mutable.Value := .scalar (.bool value)

private abbrev inputEnv (valuesAddress : Nat) : HeapCalls.Env :=
  fun name => if name = "values" then some (.ref valuesAddress) else none

private abbrev sizeRequirement : Expr := Solvers.P54Ast.requires__score_values_0

private theorem length_test (heap : Heap) (env : HeapCalls.Env) (address : Nat) (values : List Int)
    (getValues : env "values" = some (.ref address))
    (found : heap[address]? = some (.list (numbers values))) :
    Evaluates program env sizeRequirement heap (flag (decide (values.length = 5))) heap := by
  apply Evaluates.compare (av := number values.length) (bv := number 5) (middle := heap) (afterRight := heap)
  · apply Evaluates.builtin (.cons (Evaluates.lookup getValues) .nil)
    simp [primitive, Mutable.length, found, numbers]
  · exact Evaluates.constant
  · simp [HeapCalls.compare, scalarCompare, evalCmpOp, flag]
    omega

theorem «Returns» (heap : Heap) (valuesAddress suitsAddress : Nat) (_valid : heap.Valid)
    (typedValues : Mutable.Value.HasType heap (.ref valuesAddress) "list[int]")
    (typedSuits : Mutable.Value.HasType heap (.ref suitsAddress) "list[str]")
    (requires : HeapCalls.Condition program (inputEnv valuesAddress) sizeRequirement heap) :
    (SetCalls.call program "_score_values" [.ref valuesAddress, .ref suitsAddress] heap).Returns := by
  obtain ⟨values, valuesFound⟩ := Mutable.Value.ints_of_hasType typedValues
  obtain ⟨suits, suitsFound⟩ := Mutable.Value.strings_of_hasType typedSuits
  have five : values.length = 5 := by
    simpa [flag] using (length_test heap (inputEnv valuesAddress) valuesAddress values rfl valuesFound).condition_iff.mp requires
  exact SetCalls.returns_of_heap
    (numeric_score.totality.returns heap valuesAddress suitsAddress values suits [] five valuesFound suitsFound)

theorem «Supported» (heap : Heap) (valuesAddress suitsAddress : Nat) (valid : heap.Valid)
    (typedValues : Mutable.Value.HasType heap (.ref valuesAddress) "list[int]")
    (typedSuits : Mutable.Value.HasType heap (.ref suitsAddress) "list[str]")
    (requires : HeapCalls.Condition program (inputEnv valuesAddress) sizeRequirement heap) :
    (SetCalls.call program "_score_values" [.ref valuesAddress, .ref suitsAddress] heap).Supported := by
  intro choices
  exact ((«Returns» heap valuesAddress suitsAddress valid typedValues typedSuits requires) choices).supported

end solvers.«54»._score_values

namespace solvers.«54»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls

set_option linter.style.nameCheck false in
theorem requires__score_values_0_supported (heap : Heap) (valuesAddress suitsAddress : Nat) (_valid : heap.Valid)
    (typedValues : Mutable.Value.HasType heap (.ref valuesAddress) "list[int]")
    (_typedSuits : Mutable.Value.HasType heap (.ref suitsAddress) "list[str]") :
    (HeapCalls.evaluate Solvers.P54Ast.module (_score_values.inputEnv valuesAddress)
      Solvers.P54Ast.requires__score_values_0 heap).Supported := by
  obtain ⟨values, found⟩ := Mutable.Value.ints_of_hasType typedValues
  exact Execution.supported_of_finishes
    (_score_values.length_test heap (_score_values.inputEnv valuesAddress) valuesAddress values rfl found).finishes

end solvers.«54»

namespace solvers.«54».hand_rank.totality
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
set_option Elab.async false

private abbrev program := Solvers.P54Ast.module

/-- The public wrapper parses all five records through the live dictionary, then executes the numeric scorer. -/
theorem five_result (heap : Heap) (cardsAddress : Nat) (cards : List Card)
    (cardsFound : heap[cardsAddress]? = some (.list (tokens cards)))
    (five : cards.length = 5)
    (globals : Mutable.Globals) (tableAddress : Nat) (entries : List (Constant × Mutable.Value))
    (getTable : globals.lookup "VALUE_MAP" = some (.ref tableAddress))
    (tableFound : heap[tableAddress]? = some (.dict entries)) (ranked : Ranked entries cards) :
    ∃ answer finalHeap, ReturnsValue program "hand_rank" [.ref cardsAddress] heap answer finalHeap globals := by
  let ranks : List Int := cards.map Card.value
  let suitNames : List String := cards.map fun card => String.singleton card.suitChar
  have ranksFive : ranks.length = 5 := by simpa only [ranks, List.length_map] using five
  have ranksFound : (parsedHeap heap cards)[heap.length]? =
      some (.list (ranks.map fun value => .scalar (.int value))) := by
    simpa [ranks, values, List.map_map] using values_found heap cards
  have suitNamesFound : (parsedHeap heap cards)[heap.length+1]? =
      some (.list (suitNames.map fun value => .scalar (.string value))) := by
    simpa [suitNames, suits, List.map_map] using suits_found heap cards
  obtain ⟨answer, finalHeap, scored⟩ := numeric_score.totality.result (parsedHeap heap cards)
    heap.length (heap.length+1) ranks suitNames globals ranksFive ranksFound suitNamesFound
  exact ⟨answer, finalHeap, result heap cardsAddress cards cardsFound globals tableAddress entries
    getTable tableFound ranked answer finalHeap scored⟩

theorem five_returns (heap : Heap) (cardsAddress : Nat) (cards : List Card)
    (cardsFound : heap[cardsAddress]? = some (.list (tokens cards)))
    (five : cards.length = 5)
    (globals : Mutable.Globals) (tableAddress : Nat) (entries : List (Constant × Mutable.Value))
    (getTable : globals.lookup "VALUE_MAP" = some (.ref tableAddress))
    (tableFound : heap[tableAddress]? = some (.dict entries)) (ranked : Ranked entries cards) :
    (HeapCalls.call program "hand_rank" [.ref cardsAddress] heap globals).Returns := by
  obtain ⟨answer, finalHeap, executed⟩ := five_result heap cardsAddress cards cardsFound five globals
    tableAddress entries getTable tableFound ranked
  exact executed.returns

theorem five_supported (heap : Heap) (cardsAddress : Nat) (cards : List Card)
    (cardsFound : heap[cardsAddress]? = some (.list (tokens cards)))
    (five : cards.length = 5)
    (globals : Mutable.Globals) (tableAddress : Nat) (entries : List (Constant × Mutable.Value))
    (getTable : globals.lookup "VALUE_MAP" = some (.ref tableAddress))
    (tableFound : heap[tableAddress]? = some (.dict entries)) (ranked : Ranked entries cards) :
    (HeapCalls.call program "hand_rank" [.ref cardsAddress] heap globals).Supported :=
  (five_returns heap cardsAddress cards cardsFound five globals tableAddress entries getTable tableFound ranked).supported

end solvers.«54».hand_rank.totality

namespace solvers.«54».numeric_score.frame
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open solvers.«54».numeric_score.totality
set_option Elab.async false

private abbrev program := Solvers.P54Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
private abbrev numbers (values : List Int) : List Mutable.Value := values.map number
private abbrev flag (value : Bool) : Mutable.Value := .scalar (.bool value)
private def code : List Stmt := executableBody ((findFunctionIn program.body "_score_values").getD ([], [])).2

private theorem simple_frame {env : HeapCalls.Env} {body : List Stmt} {heap : Heap} {answer : Mutable.Value}
    {values : List Int} {flush straight : Bool} {high : Int} {globals : Mutable.Globals}
    (run : BlockReturns program env body heap answer (simple_categories.output heap values flush straight high) globals) :
    ∃ answer extra, BlockReturns program env body heap answer (heap ++ extra) globals := by
  cases flush <;> cases straight <;> exact ⟨answer, _, run⟩

private theorem append_frame {env : HeapCalls.Env} {body : List Stmt} {heap : Heap} {answer : Mutable.Value}
    {first second : Heap} {globals : Mutable.Globals}
    (run : BlockReturns program env body heap answer ((heap ++ first) ++ second) globals) :
    ∃ answer extra, BlockReturns program env body heap answer (heap ++ extra) globals := by
  exact ⟨answer, first ++ second, by simpa only [List.append_assoc] using run⟩

/-- All seven actual five-value profiles select a proved executable category path. -/
theorem suffix_total (heap : Heap) (env : HeapCalls.Env) (valuesAddress : Nat)
    (values : List Int) (high : Int) (flush straight : Bool) (globals : Mutable.Globals)
    (five : values.length = 5)
    (getValues : env "values" = some (.ref valuesAddress))
    (valuesFound : (straight_detection.groupedHeap heap values)[valuesAddress]? = some (.list (numbers values)))
    (getGroups : env "groups" = some (.ref (groupsAddress heap values)))
    (getFlush : env "is_flush" = some (flag flush))
    (getStraight : env "is_straight" = some (flag straight))
    (getHigh : env "straight_high" = some (number high)) :
    ∃ answer extra, BlockReturns program env (code.drop 9)
      (straight_detection.groupedHeap heap values) answer (straight_detection.groupedHeap heap values ++ extra) globals := by
  have nonempty : values ≠ [] := by intro empty; subst values; simp at five
  have listFound := straight_detection.grouped_found heap values
  have profile := straight_detection.grouped_profile heap values five
  simp only [List.map_map] at profile
  change frequencies heap values ∈ ProjectEulerSolutions.PokerCountPatterns.profiles at profile
  simp only [ProjectEulerSolutions.PokerCountPatterns.profiles, List.mem_cons, List.not_mem_nil, or_false] at profile
  rcases profile with allEqual | quad | fullHouse | triple | twoPairs | onePair | singles
  · obtain ⟨address, rank, first, pair, counted⟩ := group_at heap values 0 5 (by rw [allEqual]; rfl)
    exact simple_frame (simple_categories.result (straight_detection.groupedHeap heap values) env
      valuesAddress (groupsAddress heap values) address values (groups heap values) 5 rank high flush straight globals
      nonempty getValues valuesFound getGroups listFound first pair getFlush getStraight getHigh
      (by decide) (by decide) (by decide))

  · obtain ⟨address, rank, first, pair, counted⟩ := group_at heap values 0 4 (by rw [quad]; rfl)
    have remains : values.filter (fun value => value != rank) ≠ [] :=
      Kickers.four_kicker_nonempty values rank five (by omega)
    exact append_frame (repeated_categories.quad_path.four_of_a_kind (straight_detection.groupedHeap heap values) env
      valuesAddress (groupsAddress heap values) address values (groups heap values) rank globals
      getValues getGroups valuesFound listFound first pair remains)
  · obtain ⟨firstAddress, trip, first, tripFound, tripCount⟩ := group_at heap values 0 3 (by rw [fullHouse]; rfl)
    obtain ⟨secondAddress, pair, second, pairFound, pairCount⟩ := group_at heap values 1 2 (by rw [fullHouse]; rfl)
    exact ⟨_, _, repeated_categories.full_house (straight_detection.groupedHeap heap values) env
      (groupsAddress heap values) firstAddress secondAddress (groups heap values) trip pair globals
      getGroups listFound first second tripFound pairFound⟩
  · obtain ⟨firstAddress, rank, first, firstFound, firstCount⟩ := group_at heap values 0 3 (by rw [triple]; rfl)
    obtain ⟨secondAddress, secondRank, second, secondFound, secondCount⟩ := group_at heap values 1 1 (by rw [triple]; rfl)
    by_cases priority : flush || straight = true
    · exact simple_frame (priority_categories.triple_without_full_house (straight_detection.groupedHeap heap values) env
        valuesAddress (groupsAddress heap values) firstAddress secondAddress values (groups heap values)
        rank 1 secondRank high flush straight globals nonempty getValues valuesFound getGroups listFound
        first second firstFound secondFound getFlush getStraight getHigh (by decide) priority)
    · have noFlush : flush = false := by cases flush <;> cases straight <;> simp_all
      have noStraight : straight = false := by cases flush <;> cases straight <;> simp_all
      have remains : filtered_categories.filtered values rank ≠ [] :=
        filtered_categories.nonempty_three values rank five (by omega)
      exact ⟨_, _, filtered_categories.trips (straight_detection.groupedHeap heap values) env
        valuesAddress (groupsAddress heap values) firstAddress secondAddress values (groups heap values) rank 1 secondRank globals
        getValues valuesFound getGroups listFound first second firstFound secondFound (by decide)
        (by simpa only [noFlush] using getFlush) (by simpa only [noStraight] using getStraight) remains⟩
  · obtain ⟨firstAddress, left, first, firstFound, firstCount⟩ := group_at heap values 0 2 (by rw [twoPairs]; rfl)
    obtain ⟨secondAddress, right, second, secondFound, secondCount⟩ := group_at heap values 1 2 (by rw [twoPairs]; rfl)
    by_cases priority : flush || straight = true
    · exact simple_frame (priority_categories.without_triple_or_quad (straight_detection.groupedHeap heap values) env
        valuesAddress (groupsAddress heap values) firstAddress values (groups heap values) 2 left high flush straight globals
        nonempty getValues valuesFound getGroups listFound first firstFound getFlush getStraight getHigh
        (by decide) (by decide) priority)
    · have noFlush : flush = false := by cases flush <;> cases straight <;> simp_all
      have noStraight : straight = false := by cases flush <;> cases straight <;> simp_all
      have remains : two_pairs.remaining values left right ≠ [] := by
        rw [two_pairs.remaining_eq]
        apply Kickers.two_pair_positions_kicker_nonempty (straight_detection.countedHeap heap values) heap.length values
          (by simp [straight_detection.countedHeap]) five 0 1 (.ref firstAddress) (.ref secondAddress)
          first second (by decide) left right
        · change ProjectEulerSolutions.TupleSorting.readInts (straight_detection.groupedHeap heap values) (.ref firstAddress) = [2, left]
          simp [ProjectEulerSolutions.TupleSorting.readInts, firstFound]
        · change ProjectEulerSolutions.TupleSorting.readInts (straight_detection.groupedHeap heap values) (.ref secondAddress) = [2, right]
          simp [ProjectEulerSolutions.TupleSorting.readInts, secondFound]
      exact ⟨_, _, two_pairs.result (straight_detection.groupedHeap heap values) env
        valuesAddress (groupsAddress heap values) firstAddress secondAddress values (groups heap values) left right globals
        getValues valuesFound getGroups listFound first second firstFound secondFound
        (by simpa only [noFlush] using getFlush) (by simpa only [noStraight] using getStraight) remains⟩
  · obtain ⟨firstAddress, rank, first, firstFound, firstCount⟩ := group_at heap values 0 2 (by rw [onePair]; rfl)
    obtain ⟨secondAddress, secondRank, second, secondFound, secondCount⟩ := group_at heap values 1 1 (by rw [onePair]; rfl)
    by_cases priority : flush || straight = true
    · exact simple_frame (priority_categories.without_triple_or_quad (straight_detection.groupedHeap heap values) env
        valuesAddress (groupsAddress heap values) firstAddress values (groups heap values) 2 rank high flush straight globals
        nonempty getValues valuesFound getGroups listFound first firstFound getFlush getStraight getHigh
        (by decide) (by decide) priority)
    · have noFlush : flush = false := by cases flush <;> cases straight <;> simp_all
      have noStraight : straight = false := by cases flush <;> cases straight <;> simp_all
      have remains : filtered_categories.filtered values rank ≠ [] :=
        filtered_categories.nonempty_pair values rank five (by omega)
      exact ⟨_, _, filtered_categories.pair (straight_detection.groupedHeap heap values) env
        valuesAddress (groupsAddress heap values) firstAddress secondAddress values (groups heap values) rank 1 secondRank globals
        getValues valuesFound getGroups listFound first second firstFound secondFound (by decide)
        (by simpa only [noFlush] using getFlush) (by simpa only [noStraight] using getStraight) remains⟩
  · obtain ⟨address, rank, first, pair, counted⟩ := group_at heap values 0 1 (by rw [singles]; rfl)
    exact simple_frame (simple_categories.result (straight_detection.groupedHeap heap values) env
      valuesAddress (groupsAddress heap values) address values (groups heap values) 1 rank high flush straight globals
      nonempty getValues valuesFound getGroups listFound first pair getFlush getStraight getHigh
      (by decide) (by decide) (by decide))


theorem prepared_extends (heap : Heap) (address : Nat) (values : List Int) (suits : List String) :
    ∃ extra, composition.preparedHeap heap address values suits =
      heap.set address (.list (numbers (descending values))) ++ extra := by
  refine ⟨[.set (suitKeys suits), .set ((unique (descending values)).map Constant.int),
    .list (numbers (orderedUnique values))] ++ straight_detection.extra (orderedUnique values), ?_⟩
  simp only [composition.preparedHeap, normalizationHeap, List.append_assoc]

theorem grouped_extends (heap : Heap) (values : List Int) :
    ∃ extra, straight_detection.groupedHeap heap values = heap ++ extra := by
  refine ⟨[.dict (count_values.entries (count_values.counts values))] ++
    (.list (rank_groups.references ((straight_detection.countedHeap heap values).length+1)
      (count_values.counts values)) :: rank_groups.scratch (count_values.counts values)) ++
    [.list (groups heap values)], ?_⟩
  change ((heap ++ [.dict (count_values.entries (count_values.counts values))]) ++
    (.list (rank_groups.references ((straight_detection.countedHeap heap values).length+1) (count_values.counts values)) ::
      rank_groups.scratch (count_values.counts values))) ++ [.list (groups heap values)] = _
  simp only [List.append_assoc]

theorem scored_extends (heap : Heap) (address : Nat) (values : List Int) (suits : List String) :
    ∃ extra, composition.scoredHeap heap address values suits =
      heap.set address (.list (numbers (descending values))) ++ extra := by
  obtain ⟨preparedExtra, preparedShape⟩ := prepared_extends heap address values suits
  obtain ⟨groupedExtra, groupedShape⟩ := grouped_extends (composition.preparedHeap heap address values suits) (descending values)
  refine ⟨preparedExtra ++ groupedExtra, ?_⟩
  change straight_detection.groupedHeap (composition.preparedHeap heap address values suits) (descending values) = _
  rw [groupedShape, preparedShape, List.append_assoc]

private theorem returns_frame {name : String} {arguments : List Mutable.Value} {heap base : Heap}
    {answer : Mutable.Value} {finalHeap : Heap} {globals : Mutable.Globals}
    (run : ReturnsValue program name arguments heap answer finalHeap globals)
    (shape : ∃ extra, finalHeap = base ++ extra) :
    ∃ answer extra, ReturnsValue program name arguments heap answer (base ++ extra) globals := by
  obtain ⟨extra, same⟩ := shape
  exact ⟨answer, extra, same ▸ run⟩

/-- Every five-integer input and arbitrary string suit list terminates through the actual source helper. -/
theorem result (heap : Heap) (valuesAddress suitsAddress : Nat) (values : List Int) (suits : List String)
    (globals : Mutable.Globals) (five : values.length = 5)
    (valuesFound : heap[valuesAddress]? = some (.list (numbers values)))
    (suitsFound : heap[suitsAddress]? = some (.list (suits.map fun value => .scalar (.string value)))) :
    ∃ answer extra, ReturnsValue program "_score_values" [.ref valuesAddress, .ref suitsAddress]
      heap answer (heap.set valuesAddress (.list (numbers (descending values))) ++ extra) globals := by
  have nonempty : values ≠ [] := by intro empty; subst values; simp at five
  by_cases both : (straight_detection.detected (orderedUnique values) && isFlush suits) = true
  · obtain ⟨straight, flush⟩ := Bool.and_eq_true_iff.mp both
    have run := straight_detection.straight_flush_result heap valuesAddress suitsAddress values suits globals
      nonempty valuesFound suitsFound straight flush
    apply returns_frame run
    obtain ⟨extra, shape⟩ := prepared_extends heap valuesAddress values suits
    refine ⟨extra ++ [.tuple [number 8, number (straight_detection.high (orderedUnique values))]], ?_⟩
    change composition.preparedHeap heap valuesAddress values suits ++
      [.tuple [number 8, number (straight_detection.high (orderedUnique values))]] = _
    rw [shape, List.append_assoc]
  · have notBoth : (straight_detection.detected (orderedUnique values) && isFlush suits) = false := Bool.eq_false_iff.mpr both
    let start := composition.entry globals valuesAddress suitsAddress
    let prepared := composition.preparedHeap heap valuesAddress values suits
    let ready := composition.scoredEnv start heap valuesAddress values suits
    have getValues : ready "values" = some (.ref valuesAddress) := by
      change straight_detection.groupedEnv (composition.preparedEnv start heap values suits) prepared
        (descending values) "values" = _
      rw [straight_detection.grouped_other _ _ _ "values" (by decide) (by decide),
        composition.prepared_other _ _ _ _ "values" (by decide) (by decide) (by decide) (by decide)]
      cases globals <;> rfl
    have getGroups : ready "groups" = some (.ref (groupsAddress prepared (descending values))) := by
      simp [ready, composition.scoredEnv, straight_detection.groupedEnv, groupsAddress, prepared, setLocal]
    have getFlush : ready "is_flush" = some (flag (isFlush suits)) := by
      change straight_detection.groupedEnv (composition.preparedEnv start heap values suits) prepared
        (descending values) "is_flush" = _
      rw [straight_detection.grouped_other _ _ _ "is_flush" (by decide) (by decide)]
      exact composition.prepared_flush start heap values suits
    have getStraight : ready "is_straight" = some (flag (straight_detection.detected (orderedUnique values))) := by
      change straight_detection.groupedEnv (composition.preparedEnv start heap values suits) prepared
        (descending values) "is_straight" = _
      rw [straight_detection.grouped_other _ _ _ "is_straight" (by decide) (by decide)]
      exact composition.prepared_straight start heap values suits
    have getHigh : ready "straight_high" = some (number (straight_detection.high (orderedUnique values))) := by
      change straight_detection.groupedEnv (composition.preparedEnv start heap values suits) prepared
        (descending values) "straight_high" = _
      rw [straight_detection.grouped_other _ _ _ "straight_high" (by decide) (by decide)]
      exact straight_detection.updated_high (normalizationEnv start heap suits) (orderedUnique values)
        (by simp [normalizationEnv, setLocal])
    have sortedFive : (descending values).length = 5 :=
      (ProjectEulerSolutions.PokerUnique.descending_perm values).length_eq.trans five
    have originalValues : prepared[valuesAddress]? = some (.list (numbers (descending values))) :=
      composition.prepared_values heap valuesAddress values suits valuesFound
    obtain ⟨answer, extra, suffix⟩ := suffix_total prepared ready valuesAddress (descending values)
      (straight_detection.high (orderedUnique values)) (isFlush suits) (straight_detection.detected (orderedUnique values))
      globals sortedFive getValues (grouped_preserves prepared (descending values) valuesAddress _ originalValues)
      getGroups getFlush getStraight getHigh
    have run := composition.result_of_suffix heap valuesAddress suitsAddress values suits globals
      nonempty valuesFound suitsFound notBoth answer (composition.scoredHeap heap valuesAddress values suits ++ extra) suffix
    apply returns_frame run
    obtain ⟨initialExtra, shape⟩ := scored_extends heap valuesAddress values suits
    exact ⟨initialExtra ++ extra, by rw [shape, List.append_assoc]⟩


end solvers.«54».numeric_score.frame

namespace solvers.«54».hand_rank.frame
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
private abbrev program := Solvers.P54Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
private abbrev numbers (values : List Int) : List Mutable.Value := values.map number

/-- The public scorer changes only its fresh parsing buffers and appends private rank objects. -/
theorem result (heap : Heap) (cardsAddress : Nat) (cards : List Card)
    (cardsFound : heap[cardsAddress]? = some (.list (tokens cards))) (five : cards.length = 5)
    (globals : Mutable.Globals) (tableAddress : Nat) (entries : List (Constant × Mutable.Value))
    (getTable : globals.lookup "VALUE_MAP" = some (.ref tableAddress))
    (tableFound : heap[tableAddress]? = some (.dict entries)) (ranked : Ranked entries cards) :
    ∃ answer extra, ReturnsValue program "hand_rank" [.ref cardsAddress] heap answer (heap ++ extra) globals := by
  let ranks : List Int := cards.map Card.value
  let suitNames : List String := cards.map fun card => String.singleton card.suitChar
  have ranksFive : ranks.length = 5 := by simpa only [ranks, List.length_map] using five
  have ranksFound : (parsedHeap heap cards)[heap.length]? =
      some (.list (ranks.map fun value => .scalar (.int value))) := by
    simpa [ranks, values, List.map_map] using values_found heap cards
  have suitNamesFound : (parsedHeap heap cards)[heap.length+1]? =
      some (.list (suitNames.map fun value => .scalar (.string value))) := by
    simpa [suitNames, suits, List.map_map] using suits_found heap cards
  obtain ⟨answer, extra, scored⟩ := numeric_score.frame.result (parsedHeap heap cards)
    heap.length (heap.length+1) ranks suitNames globals ranksFive ranksFound suitNamesFound
  have mutated : (parsedHeap heap cards).set heap.length (.list (numbers (numeric_score.descending ranks))) =
      heap ++ ([.list (numbers (numeric_score.descending ranks)), .list (suits cards)] ++ pairs cards) := by
    simp [parsedHeap, List.append_assoc, List.set_append_right]
  have run := hand_rank.result heap cardsAddress cards cardsFound globals tableAddress entries getTable tableFound ranked
    answer _ scored
  rw [mutated] at run
  exact ⟨answer, _, by simpa only [List.append_assoc] using run⟩

end solvers.«54».hand_rank.frame

namespace ProjectEulerSolutions.PokerFlushEquivalence
open FormalSW.PyAst
set_option Elab.async false

def addUnique (values : List String) (value : String) : List String :=
  if value ∈ values then values else values ++ [value]

def unique (values : List String) : List String := values.foldl addUnique []

/-- Homogeneous string membership uses ordinary string equality. -/
theorem contains_strings (values : List String) (value : String) :
    ScalarSet.contains (values.map Constant.string) (.string value) = decide (value ∈ values) := by
  induction values with
  | nil => rfl
  | cons first rest ih =>
      change ((scalarCompare .eq (.string first) (.string value)).getD false ||
        ScalarSet.contains (rest.map Constant.string) (.string value)) = _
      rw [ih]
      simp [scalarCompare, eq_comm]

/-- String insertion preserves the first representative, just as the source set constructor does. -/
theorem insert_strings (values : List String) (value : String) :
    ScalarSet.insert (values.map Constant.string) (.string value) = (addUnique values value).map Constant.string := by
  simp only [ScalarSet.insert, contains_strings, addUnique, decide_eq_true_eq]
  split <;> simp_all

theorem fold_strings (values before : List String) :
    (values.map Constant.string).foldl ScalarSet.insert (before.map Constant.string) =
      (values.foldl addUnique before).map Constant.string := by
  induction values generalizing before with
  | nil => rfl
  | cons first rest ih =>
      change (rest.map Constant.string).foldl ScalarSet.insert (ScalarSet.insert (before.map Constant.string) (.string first)) =
        (rest.foldl addUnique (addUnique before first)).map Constant.string
      rw [insert_strings]
      exact ih (addUnique before first)

theorem mem_addUnique (query : String) (values : List String) (value : String) :
    query ∈ addUnique values value ↔ query ∈ values ∨ query = value := by
  by_cases present : value ∈ values
  · simp only [addUnique, if_pos present]
    constructor
    · exact Or.inl
    · rintro (member | rfl)
      · exact member
      · exact present
  · simp [addUnique, present]

/-- Folding cannot discard old members and introduces precisely the supplied strings. -/
theorem mem_fold (query : String) (values before : List String) :
    query ∈ values.foldl addUnique before ↔ query ∈ before ∨ query ∈ values := by
  induction values generalizing before with
  | nil => simp
  | cons first rest ih =>
      rw [List.foldl_cons, ih, mem_addUnique]
      simp only [List.mem_cons]
      exact or_assoc

theorem mem_unique (query : String) (values : List String) :
    query ∈ unique values ↔ query ∈ values := by
  simpa only [unique, List.not_mem_nil, false_or] using mem_fold query values []

theorem fold_singleton (values : List String) (first : String)
    (same : ∀ value ∈ values, value = first) : values.foldl addUnique [first] = [first] := by
  induction values with
  | nil => rfl
  | cons value rest ih =>
      have head : value = first := same value (by simp)
      subst value
      have tail : ∀ value ∈ rest, value = first := by
        intro value member
        exact same value (by simp [member])
      simpa only [List.foldl_cons, addUnique, List.mem_singleton, if_pos rfl] using ih tail

theorem unique_cons_eq_singleton (first : String) (tail : List String)
    (same : ∀ value ∈ tail, value = first) : unique (first :: tail) = [first] := by
  change tail.foldl addUnique (addUnique [] first) = [first]
  simpa only [addUnique, List.not_mem_nil, if_false, List.nil_append] using fold_singleton tail first same

private abbrev referenceFlush :=
  data.«project-euler-statements».data.py.project_euler_statements.p54.flush_source.flush

/-- A single retained suit is equivalent to the nonempty reference tail-equality test. -/
theorem unique_length_one_iff (values : List String) :
    (unique values).length = 1 ↔ referenceFlush values = true := by
  cases values with
  | nil => simp [unique, referenceFlush, data.«project-euler-statements».data.py.project_euler_statements.p54.flush_source.flush]
  | cons first tail =>
      change (unique (first :: tail)).length = 1 ↔ (tail.all (fun value => decide (value = first))) = true
      rw [List.all_eq_true]
      simp only [decide_eq_true_eq]
      constructor
      · intro one
        obtain ⟨only, singleton⟩ := List.length_eq_one_iff.mp one
        have firstMember : first ∈ unique (first :: tail) := (mem_unique first _).mpr (by simp)
        have onlyEq : first = only := by simpa only [singleton, List.mem_singleton] using firstMember
        subst only
        intro value member
        have retained : value ∈ unique (first :: tail) := (mem_unique value _).mpr (by simp [member])
        simpa only [singleton, List.mem_singleton] using retained
      · intro same
        rw [unique_cons_eq_singleton first tail same]
        rfl

theorem suitKeys_eq (values : List String) :
    solvers.«54».numeric_score.suitKeys values = (unique values).map Constant.string := by
  exact fold_strings values []

/-- The two checked implementations compute the same flush flag for every string list. -/
theorem isFlush_eq (values : List String) :
    solvers.«54».numeric_score.isFlush values =
      data.«project-euler-statements».data.py.project_euler_statements.p54.flush_source.flush values := by
  rw [solvers.«54».numeric_score.isFlush, suitKeys_eq]
  simp only [List.length_map]
  apply Bool.eq_iff_iff.mpr
  simp only [decide_eq_true_eq]
  exact unique_length_one_iff values

end ProjectEulerSolutions.PokerFlushEquivalence
