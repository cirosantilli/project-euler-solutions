import solvers.«98.py.lean.d».Grouping
import lang.py.FormalSW.DictionaryValuesCalls

namespace solvers.«98».groups
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
open StringValueBuckets (Bucket View)
open pattern_signature (program number)
set_option Elab.async false

def reference (bucket : Bucket) : Mutable.Value := .ref bucket.address
def fields (buckets : List Bucket) : List (Mutable.Value × Mutable.Value) :=
  buckets.map fun bucket => (.scalar (.string bucket.key), reference bucket)
def keys (buckets : List Bucket) : List SequenceSource := (fields buckets).map fun entry => sequenceSource entry.1
def selected (buckets : List Bucket) : List Bucket := buckets.filter fun bucket => decide (2 ≤ bucket.values.length)
abbrev test : Expr := .compare (.builtinCall .len [.name "g"]) [(.gtE, .constant (.int 2))]
abbrev body (collector : Nat) : List Stmt := comprehensionFilters [test] [.comprehensionAppend collector (.name "g")]
abbrev expression : Expr := .listCompTargets (.name "g") [(.name "g", .dictionaryValuesIterable (.name "groups"), [test])]
abbrev work (base : Mutable.Heap) (output : List Bucket) (garbage : Mutable.Heap) :=
  HeapCollection.working base (output.map reference) garbage

theorem source : (((findFunctionIn program.body "solve").getD ([], [])).2)[2]? =
    some (.assign [.name "anagram_groups"] expression) := rfl

theorem dictionary_fields (buckets : List Bucket) :
    dictionaryItemFields (StringValueBuckets.object buckets) = some (fields buckets) := by
  simp [dictionaryItemFields, StringValueBuckets.object, StringValueBuckets.entries, StringValueBuckets.entry,
    fields, reference, List.map_map, Function.comp_def]

theorem View.next {heap : Mutable.Heap} {dictionary : Nat} {buckets : List Bucket}
    (view : View heap dictionary buckets) (position : Nat) :
    dictionaryItemsNext dictionary (keys buckets) position heap = some (.ok ((fields buckets)[position]?), heap) :=
  dictionaryItemsNext_fields view.found (dictionary_fields buckets)

theorem selected_cons (bucket : Bucket) (buckets : List Bucket) :
    selected (bucket :: buckets) = (if 2 ≤ bucket.values.length then [bucket] else []) ++ selected buckets := by
  by_cases accepted : 2 ≤ bucket.values.length <;> simp [selected, accepted]

theorem collect (bucket : Bucket) (base : Mutable.Heap) (output : List Bucket) (garbage : Mutable.Heap)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (found : base[bucket.address]? = some (.list bucket.values)) (getGroup : env "g" = some (reference bucket)) :
    BlockContinues program env (body base.length) (work base output garbage) env
      (work base (output ++ if 2 ≤ bucket.values.length then [bucket] else []) garbage) globals := by
  have kept := HeapCollection.prefix_found base (.list (output.map reference) :: garbage) bucket.address _ found
  have tested : Evaluates program env test (work base output garbage) (.scalar (.bool (decide (2 ≤ bucket.values.length))))
      (work base output garbage) globals := by
    apply Evaluates.compare (av := number bucket.values.length) (bv := number 2)
      (middle := work base output garbage) (afterRight := work base output garbage)
    · apply Evaluates.builtin (.cons (Evaluates.lookup getGroup) .nil)
      simp [primitive, Mutable.length, reference, kept, work, HeapCollection.working, number]
    · exact Evaluates.constant
    · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
      norm_cast
  by_cases accepted : 2 ≤ bucket.values.length
  · simp only [if_pos accepted]
    apply BlockContinues.branch (flag := true) tested (by simp [Mutable.truth, scalarTruth, accepted])
    have appended := BlockContinues.comprehension_append (module := program) (globals := globals) (Evaluates.lookup getGroup)
      (HeapCollection.found base (output.map reference) garbage)
    simpa only [work, HeapCollection.store, List.map_append, List.map_cons, List.map_nil] using appended
  · simp only [if_neg accepted, List.append_nil]
    exact BlockContinues.branch (flag := false) tested (by simp [Mutable.truth, scalarTruth, accepted]) BlockContinues.nil

theorem loop (base : Mutable.Heap) (dictionary : Nat) (buckets : List Bucket) (view : View base dictionary buckets)
    (remaining : List Bucket) (position : Nat) (suffix : buckets.drop position = remaining)
    (output : List Bucket) (garbage : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals) :
    ∃ finalEnv finalGarbage, BlockContinues program env
      [.dictionaryItemsStepLoop "$formalsw.values.pair" dictionary (keys buckets) position
        (dictionaryValuesBody (.name "g") (body base.length)) []]
      (work base output garbage) finalEnv (work base (output ++ selected remaining) finalGarbage) globals := by
  induction remaining generalizing position output garbage env with
  | nil =>
      refine ⟨env, garbage, ?_⟩
      simp only [selected, List.filter_nil, List.append_nil]
      apply BlockContinues.dictionary_items_done
      · have missing : buckets[position]? = none := by
          simpa [suffix] using (List.head?_drop (l := buckets) (i := position)).symm
        simpa only [fields, List.getElem?_map, missing, Option.map_none] using
          View.next (view.append (.list (output.map reference) :: garbage)) position
      · exact BlockContinues.nil
  | cons bucket rest ih =>
      have item : buckets[position]? = some bucket := by
        simpa [suffix] using (List.head?_drop (l := buckets) (i := position)).symm
      have member := List.mem_of_getElem? item
      let current := work base output garbage
      let pair : Mutable.Object := .tuple [.scalar (.string bucket.key), reference bucket]
      let afterPair := work base output (garbage ++ [pair])
      let paired := setLocal env "$formalsw.values.pair" (.ref current.length)
      let named := assignNames paired ["$formalsw.values.key", "$formalsw.values.value"] [.scalar (.string bucket.key), reference bucket]
      let ready := setLocal named "g" (reference bucket)
      have layout : current ++ [pair] = afterPair := HeapCollection.append base (output.map reference) garbage [pair]
      have pairFound : afterPair[current.length]? = some pair := by rw [← layout]; simp
      have unpacked : BlockContinues program paired
          [.assign [.tuple [.name "$formalsw.values.key", .name "$formalsw.values.value"]] (.name "$formalsw.values.pair")]
          afterPair named afterPair globals := by
        apply BlockContinues.unpack (names := ["$formalsw.values.key", "$formalsw.values.value"])
          (items := [.scalar (.string bucket.key), reference bucket]) (value := .ref current.length)
          (middle := afterPair) (finalHeap := afterPair) rfl (Evaluates.lookup (by simp [paired, setLocal]))
          (by simp [Mutable.sequence, pairFound, pair]) rfl
      have assigned : BlockContinues program named [.assign [.name "g"] (.name "$formalsw.values.value")]
          afterPair ready afterPair globals := BlockContinues.assign (Evaluates.lookup (by simp [named, assignNames, setLocal]))
      have filtered := collect bucket base output (garbage ++ [pair]) ready globals (view.lists bucket member)
        (by simp [ready, setLocal])
      have executed := unpacked.append (assigned.append filtered)
      have tailSuffix : buckets.drop (position + 1) = rest := by
        simpa [List.drop_drop, Nat.add_comm] using congrArg (List.drop 1) suffix
      obtain ⟨finalEnv, finalGarbage, continued⟩ := ih (position + 1) tailSuffix
        (output ++ if 2 ≤ bucket.values.length then [bucket] else []) (garbage ++ [pair]) ready
      refine ⟨finalEnv, finalGarbage, ?_⟩
      apply BlockContinues.dictionary_items_step
      · simpa only [fields, List.getElem?_map, item, Option.map_some] using
          View.next (view.append (.list (output.map reference) :: garbage)) position
      · have allocated : Mutable.allocateTuple [.scalar (.string bucket.key), reference bucket] current =
            some (.ok (.ref current.length), current ++ [pair]) := rfl
        simpa only [layout] using allocated
      · simpa only [Calls.loopContinuation,
          show Calls.loopControl (dictionaryValuesBody (.name "g") (body base.length)) = false from rfl,
          Bool.false_eq_true, if_false, selected_cons, List.append_assoc]
          using executed.append continued

theorem result (heap : Mutable.Heap) (dictionary : Nat) (buckets : List Bucket) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (view : View heap dictionary buckets) (getGroups : env "groups" = some (.ref dictionary)) :
    ∃ garbage, Evaluates program env expression heap (.ref heap.length) (work heap (selected buckets) garbage) globals := by
  let inner := comprehensionTargetsEnv env [(.name "g", .dictionaryValuesIterable (.name "groups"), [test])]
  obtain ⟨finalEnv, garbage, executed⟩ := loop heap dictionary buckets view buckets 0 (by simp) [] [] inner globals
  refine ⟨garbage, ?_⟩
  apply Evaluates.list_comp_dictionary_values (finalEnv := finalEnv) (Evaluates.lookup getGroups) view.found (dictionary_fields buckets)
  simpa only [inner, body, comprehensionTargetsBody, List.nil_append, work, List.map_nil] using executed

/-- The first three source statements construct and filter the complete anagram partition. -/
theorem prefix_result (words : List String) (heap : Mutable.Heap) (wordsAddress : Nat) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (found : heap[wordsAddress]? = some (.list (words.map solve.text))) (getWords : env "words" = some (.ref wordsAddress)) :
    ∃ finalEnv extra buckets address,
      BlockContinues program env (((findFunctionIn program.body "solve").getD ([], [])).2.take 3) heap finalEnv (heap ++ extra) globals ∧
      View (heap ++ extra) heap.length buckets ∧
      StringValueBuckets.Partition solve.keyOfValue (solve.groupValues words) buckets ∧
      (heap ++ extra)[address]? = some (.list ((selected buckets).map reference)) ∧
      finalEnv "groups" = some (.ref heap.length) ∧ finalEnv "anagram_groups" = some (.ref address) ∧
      finalEnv "words" = some (.ref wordsAddress) := by
  obtain ⟨groupEnv, suffix, buckets, grouped, view, partition, getGroups, input⟩ :=
    solve.grouping_result words heap wordsAddress env globals found getWords
  obtain ⟨garbage, filtered⟩ := result (heap ++ suffix) heap.length buckets groupEnv globals view getGroups
  let finalEnv := setLocal groupEnv "anagram_groups" (.ref (heap ++ suffix).length)
  refine ⟨finalEnv, suffix ++ (.list ((selected buckets).map reference) :: garbage), buckets, (heap ++ suffix).length,
    ?_, ?_, partition, ?_, ?_, ?_, ?_⟩
  · simpa only [work, HeapCollection.working, List.append_assoc] using grouped.append (BlockContinues.assign filtered)
  · simpa only [List.append_assoc] using view.append (.list ((selected buckets).map reference) :: garbage)
  · simpa only [work, HeapCollection.working, List.append_assoc] using HeapCollection.found (heap ++ suffix) ((selected buckets).map reference) garbage
  · simpa [finalEnv, setLocal] using getGroups
  · simp [finalEnv, setLocal]
  · simpa [finalEnv, setLocal] using input

end solvers.«98».groups
