import solvers.«98.py.lean.d».SquareBounds
import data.«project-euler-statements».data.py.project_euler_statements.«p98.py»
import ProjectEulerSolutions.Lib.StringValueBuckets

namespace solvers.«98».solve
open FormalSW.PyAst HeapCalls ProjectEulerSolutions ProjectEulerSolutions.AnagramKeys
open pattern_signature (program)
set_option Elab.async false

abbrev keyExpression : Expr := .call (.attribute (.constant (.string "")) "join")
  [.builtinCall .sorted [.name "w"]]
abbrev groupingBody : List Stmt := [.ifThen (.name "w")
    [.expr (.call (.attribute (.call (.attribute (.name "groups") "setdefault")
      [keyExpression, .list []]) "append") [.name "w"])] []]
abbrev grouping : Stmt := .forIn (.name "w") (.name "words") groupingBody []

theorem source_grouping :
    (((findFunctionIn program.body "solve").getD ([], [])).2.drop 1).take 1 = [grouping] := rfl

theorem key_result (word : String) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (globals : Mutable.Globals) (getWord : env "w" = some (.scalar (.string word))) :
    Evaluates program env keyExpression heap (.scalar (.string (key word)))
      (heap ++ [.list (values word)]) globals :=
  evaluate_key program env heap (.name "w") word globals (Evaluates.lookup getWord)

/-- The reference accepts exactly the distinct words in the optimized dictionary's same key. -/
theorem reference_filter (word other : String) (address : Nat) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (globals : Mutable.Globals)
    (getWord : env "w" = some (.scalar (.string word)))
    (getOther : env "u" = some (.scalar (.string other)))
    (getKey : env "key" = some (.ref address))
    (found : heap[address]? = some (.list (values word))) :
    ∃ extra, Evaluates P98Ast.module env
      data.«project-euler-statements».data.py.project_euler_statements.p98.anagrams.condition heap
      (.scalar (.bool (decide (word ≠ other ∧ key word = key other)))) (heap ++ extra) globals :=
  data.«project-euler-statements».data.py.project_euler_statements.p98.anagrams.condition_result
    word other address env heap globals getWord getOther getKey found

abbrev text (word : String) : Mutable.Value := .scalar (.string word)
def keyOfValue : Mutable.Value → String
  | .scalar (.string word) => key word
  | _ => ""
def groupValues (words : List String) : List Mutable.Value := (words.filter (· ≠ "")).map text

theorem grouping_step (before : List String) (word : String) (base suffix : Mutable.Heap)
    (dictionary : Nat) (buckets : List StringValueBuckets.Bucket) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (view : StringValueBuckets.View (base ++ suffix) dictionary buckets)
    (partition : StringValueBuckets.Partition keyOfValue (groupValues before) buckets)
    (afterBase : base.length ≤ dictionary) (getGroups : env "groups" = some (.ref dictionary))
    (getWord : env "w" = some (text word)) :
    ∃ newSuffix newBuckets, BlockContinues program env groupingBody (base ++ suffix) env (base ++ newSuffix) globals ∧
      StringValueBuckets.View (base ++ newSuffix) dictionary newBuckets ∧
      StringValueBuckets.Partition keyOfValue (groupValues (before ++ [word])) newBuckets := by
  by_cases empty : word = ""
  · refine ⟨suffix, buckets, ?_, view, ?_⟩
    · apply BlockContinues.branch (flag := false) (Evaluates.lookup getWord)
        (by simp [Mutable.truth, scalarTruth, empty])
      exact BlockContinues.nil
    · simpa [groupValues, List.filter_append, empty] using partition
  · let sortedHeap := (base ++ suffix) ++ [.list (values word)]
    let newBuckets := StringValueBuckets.push buckets (key word) sortedHeap.length (text word)
    have sortedView : StringValueBuckets.View sortedHeap dictionary buckets := view.append _
    obtain ⟨newSuffix, preserved⟩ := StringValueBuckets.updated_prefix base (suffix ++ [.list (values word)]) dictionary buckets
      (by simpa only [sortedHeap, List.append_assoc] using sortedView) afterBase (key word) (text word)
    have inserted := StringValueBuckets.evaluate (globals := globals) program env (base ++ suffix) dictionary buckets view
      (key word) (text word) [.list (values word)] keyExpression (.name "w") getGroups
      (key_result word env (base ++ suffix) globals getWord) (Evaluates.lookup getWord)
    have updatedView := StringValueBuckets.updated_view sortedHeap dictionary buckets sortedView (key word) (text word)
    have nextPartition := partition.push sortedHeap.length (text word)
    refine ⟨newSuffix, newBuckets, ?_, ?_, ?_⟩
    · apply BlockContinues.branch (flag := true) (Evaluates.lookup getWord)
        (by simp [Mutable.truth, scalarTruth, empty])
      have emitted := BlockContinues.discard inserted
      simpa only [StringValueBuckets.insertion, List.append_assoc, preserved] using emitted
    · simpa only [newBuckets, sortedHeap, List.append_assoc, preserved] using updatedView
    · simpa [newBuckets, keyOfValue, groupValues, List.filter_append, empty] using nextPartition

theorem grouping_loop (remaining before : List String) (base suffix : Mutable.Heap)
    (wordsAddress dictionary : Nat) (buckets : List StringValueBuckets.Bucket) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (input : base[wordsAddress]? = some (.list ((before ++ remaining).map text)))
    (view : StringValueBuckets.View (base ++ suffix) dictionary buckets)
    (partition : StringValueBuckets.Partition keyOfValue (groupValues before) buckets)
    (afterBase : base.length ≤ dictionary) (getGroups : env "groups" = some (.ref dictionary)) :
    ∃ finalEnv newSuffix newBuckets, BlockContinues program env
      [.sequenceStepLoop "w" (.ref wordsAddress) before.length groupingBody []] (base ++ suffix) finalEnv (base ++ newSuffix) globals ∧
      StringValueBuckets.View (base ++ newSuffix) dictionary newBuckets ∧
      StringValueBuckets.Partition keyOfValue (groupValues (before ++ remaining)) newBuckets ∧
      (∀ name, name ≠ "w" → finalEnv name = env name) := by
  induction remaining generalizing before suffix buckets env with
  | nil =>
      refine ⟨env, suffix, buckets, ?_, view, ?_, by intros; rfl⟩
      · apply BlockContinues.sequence_done (items := before.map text)
        · have found := HeapCollection.prefix_found base suffix wordsAddress _ input
          simp [Mutable.sequence, sequenceValue, found]
        · simp
        · exact BlockContinues.nil
      · simpa using partition
  | cons word rest ih =>
      let ready := setLocal env "w" (text word)
      obtain ⟨stepSuffix, stepBuckets, stepRan, stepView, stepPartition⟩ := grouping_step before word base suffix dictionary buckets ready globals
        view partition afterBase (by simpa [ready, setLocal] using getGroups) (by simp [ready, setLocal])
      obtain ⟨finalEnv, newSuffix, newBuckets, continued, finalView, finalPartition, kept⟩ :=
        ih (before ++ [word]) stepSuffix stepBuckets ready
          (by simpa only [List.append_assoc, List.singleton_append] using input) stepView stepPartition
          (by simpa [ready, setLocal] using getGroups)
      refine ⟨finalEnv, newSuffix, newBuckets, ?_, finalView, ?_, ?_⟩
      · apply BlockContinues.sequence_step (items := (before ++ word :: rest).map text) (value := text word)
        · have found := HeapCollection.prefix_found base suffix wordsAddress _ input
          simp [Mutable.sequence, sequenceValue, found]
        · simp
        · simpa only [groupingBody, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
            Bool.false_or, Bool.false_eq_true, if_false, List.length_append, List.length_singleton]
            using stepRan.append continued
      · simpa only [List.append_assoc, List.singleton_append] using finalPartition
      · intro name different
        rw [kept name different]
        simp [ready, setLocal, different]

/-- The unchanged grouping prefix builds precisely the nonempty words in each canonical anagram class. -/
theorem grouping_result (words : List String) (heap : Mutable.Heap) (wordsAddress : Nat)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (found : heap[wordsAddress]? = some (.list (words.map text)))
    (getWords : env "words" = some (.ref wordsAddress)) :
    ∃ finalEnv suffix buckets,
      BlockContinues program env (((findFunctionIn program.body "solve").getD ([], [])).2.take 2) heap
        finalEnv (heap ++ suffix) globals ∧
      StringValueBuckets.View (heap ++ suffix) heap.length buckets ∧
      StringValueBuckets.Partition keyOfValue (groupValues words) buckets ∧
      finalEnv "groups" = some (.ref heap.length) ∧ finalEnv "words" = some (.ref wordsAddress) := by
  let ready := setLocal env "groups" (.ref heap.length)
  obtain ⟨finalEnv, suffix, buckets, executed, view, partition, kept⟩ := grouping_loop words [] heap [.dict []]
    wordsAddress heap.length [] ready globals (by simpa using found) (StringValueBuckets.empty_view heap)
    (StringValueBuckets.Partition.empty keyOfValue) (by omega) (by simp [ready, setLocal])
  refine ⟨finalEnv, suffix, buckets, ?_, view, by simpa using partition, ?_, ?_⟩
  · apply BlockContinues.append (body := (((findFunctionIn program.body "solve").getD ([], [])).2.take 1))
      (BlockContinues.annAssign (Evaluates.dictionary .nil))
    apply BlockContinues.for_sequence (items := words.map text) (by intros; simp)
      (Evaluates.lookup (by simpa [ready, setLocal] using getWords))
    · have inputKept := HeapCollection.prefix_found heap [.dict []] wordsAddress _ found
      simp [Mutable.sequence, inputKept]
    · exact executed
  · rw [kept "groups" (by decide)]
    simp [ready, setLocal]
  · rw [kept "words" (by decide)]
    simpa [ready, setLocal] using getWords

end solvers.«98».solve
