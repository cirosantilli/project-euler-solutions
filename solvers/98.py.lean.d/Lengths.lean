import solvers.«98.py.lean.d».GroupCoverage
import lang.py.FormalSW.ComprehensionCalls

namespace solvers.«98».lengths
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
open StringValueBuckets (Bucket View)
open pattern_signature (program)
set_option Elab.async false

def wordsIn (words : List String) (bucket : Bucket) : List String :=
  (words.filter (· ≠ "")).filter fun word => decide (AnagramKeys.key word = bucket.key)

theorem bucket_values (words : List String) (buckets : List Bucket)
    (partition : StringValueBuckets.Partition solve.keyOfValue (solve.groupValues words) buckets)
    (bucket : Bucket) (member : bucket ∈ buckets) : bucket.values = (wordsIn words bucket).map solve.text := by
  rw [partition.bucket_values bucket member]
  simp [wordsIn, solve.groupValues, List.filter_map, solve.keyOfValue, Function.comp_def]
  rfl

def insert (items : List Int) (value : Int) : List Int := if value ∈ items then items else items ++ [value]
def collect (items values : List Int) : List Int := values.foldl insert items
def widths (words : List String) (buckets : List Bucket) : List Int :=
  buckets.flatMap fun bucket => (wordsIn words bucket).map fun word => (word.length : Int)
def needed (words : List String) (buckets : List Bucket) : List Int :=
  ScalarSort.ints (collect [] (widths words (groups.selected buckets)))
abbrev number (n : Int) : Mutable.Value := .scalar (.int n)
abbrev work (base : Mutable.Heap) (items : List Int) : Mutable.Heap := base ++ [.set (items.map Constant.int)]
abbrev element : Expr := .builtinCall .len [.name "w"]
abbrev innerBody (address : Nat) : List Stmt := [.comprehensionAdd address element]
abbrev outerBody (address : Nat) : List Stmt := [.forIn (.name "w") (.name "g") (innerBody address) []]
abbrev expression : Expr := .setComp element [(.name "g", .name "anagram_groups", []), (.name "w", .name "g", [])]
abbrev sortedExpression : Expr := .builtinCall .sorted [expression]

theorem source : (((findFunctionIn program.body "solve").getD ([], [])).2)[4]? =
    some (.assign [.name "lengths_needed"] sortedExpression) := rfl

theorem collect_append (items first rest : List Int) : collect items (first ++ rest) = collect (collect items first) rest := by
  simp [collect, List.foldl_append]

theorem mem_collect (items values : List Int) (value : Int) : value ∈ collect items values ↔ value ∈ items ∨ value ∈ values := by
  induction values generalizing items with
  | nil => simp [collect]
  | cons first rest ih =>
      change value ∈ collect (insert items first) rest ↔ _
      rw [ih]
      by_cases present : first ∈ items <;> simp [insert, present] <;> aesop

theorem word_loop (before remaining : List String) (base : Mutable.Heap) (address : Nat) (items : List Int)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (found : base[address]? = some (.list ((before ++ remaining).map solve.text))) :
    ∃ finalEnv, BlockContinues program env [.sequenceStepLoop "w" (.ref address) before.length (innerBody base.length) []]
      (work base items) finalEnv (work base (collect items (remaining.map fun word => (word.length : Int)))) globals := by
  induction remaining generalizing before items env with
  | nil =>
      refine ⟨env, ?_⟩
      apply BlockContinues.sequence_done (items := before.map solve.text)
      · have kept := HeapCollection.prefix_found base [.set (items.map Constant.int)] address _ found
        simp [Mutable.sequence, sequenceValue, kept]
      · simp
      · exact BlockContinues.nil
  | cons word rest ih =>
      let ready := setLocal env "w" (solve.text word)
      have measured : Evaluates program ready element (work base items) (number word.length) (work base items) globals := by
        apply Evaluates.builtin (.cons (Evaluates.lookup (show ready "w" = some (solve.text word) by simp [ready, setLocal])) .nil)
        simp [primitive, Mutable.length, solve.text, number]
      have emitted := BlockContinues.comprehension_add measured (show (work base items)[base.length]? = some (.set (items.map Constant.int)) by simp)
      have written : (work base items).set base.length (.set (ScalarSet.insert (items.map Constant.int) (.int word.length))) =
          work base (insert items word.length) := by
        simp [work, ScalarSet.insert_ints, insert, List.set_append_right]
      rw [written] at emitted
      obtain ⟨finalEnv, continued⟩ := ih (before ++ [word]) (insert items word.length) ready
        (by simpa only [List.append_assoc, List.singleton_append] using found)
      refine ⟨finalEnv, ?_⟩
      apply BlockContinues.sequence_step (items := (before ++ word :: rest).map solve.text) (value := solve.text word)
      · have kept := HeapCollection.prefix_found base [.set (items.map Constant.int)] address _ found
        simp [Mutable.sequence, sequenceValue, kept]
      · simp
      · simpa only [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, innerBody,
          Bool.false_or, Bool.false_eq_true, if_false, collect, List.map_cons, List.foldl_cons,
          List.length_append, List.length_singleton] using emitted.append continued

theorem group_loop (words : List String) (base : Mutable.Heap) (address : Nat) (before remaining : List Bucket)
    (items : List Int) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (found : base[address]? = some (.list ((before ++ remaining).map groups.reference)))
    (rows : ∀ bucket ∈ remaining, base[bucket.address]? = some (.list ((wordsIn words bucket).map solve.text))) :
    ∃ finalEnv, BlockContinues program env [.sequenceStepLoop "g" (.ref address) before.length (outerBody base.length) []]
      (work base items) finalEnv (work base (collect items (widths words remaining))) globals := by
  induction remaining generalizing before items env with
  | nil =>
      refine ⟨env, ?_⟩
      apply BlockContinues.sequence_done (items := before.map groups.reference)
      · have kept := HeapCollection.prefix_found base [.set (items.map Constant.int)] address _ found
        simp [Mutable.sequence, sequenceValue, kept]
      · simp
      · exact BlockContinues.nil
  | cons bucket rest ih =>
      let ready := setLocal env "g" (groups.reference bucket)
      have row := rows bucket (by simp)
      obtain ⟨innerEnv, scanned⟩ := word_loop [] (wordsIn words bucket) base bucket.address items ready globals (by simpa using row)
      have looped : BlockContinues program ready (outerBody base.length) (work base items) innerEnv
          (work base (collect items ((wordsIn words bucket).map fun word => (word.length : Int)))) globals := by
        apply BlockContinues.for_sequence (value := groups.reference bucket) (items := (wordsIn words bucket).map solve.text) (by intros; simp)
          (Evaluates.lookup (by simp [ready, setLocal]))
        · have kept := HeapCollection.prefix_found base [.set (items.map Constant.int)] bucket.address _ row
          simp [Mutable.sequence, groups.reference, kept]
        · exact scanned
      obtain ⟨finalEnv, continued⟩ := ih (before ++ [bucket])
        (collect items ((wordsIn words bucket).map fun word => (word.length : Int))) innerEnv
        (by simpa only [List.append_assoc, List.singleton_append] using found)
        (fun other member => rows other (by simp [member]))
      refine ⟨finalEnv, ?_⟩
      apply BlockContinues.sequence_step (items := (before ++ bucket :: rest).map groups.reference) (value := groups.reference bucket)
      · have kept := HeapCollection.prefix_found base [.set (items.map Constant.int)] address _ found
        simp [Mutable.sequence, sequenceValue, kept]
      · simp
      · simpa only [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, outerBody, innerBody,
          Bool.false_or, Bool.false_eq_true, if_false, widths, List.flatMap_cons, collect_append,
          List.length_append, List.length_singleton] using looped.append continued

/-- The nested set comprehension traverses the actual bucket lists and deduplicates their lengths. -/
theorem set_result (words : List String) (buckets : List Bucket) (heap : Mutable.Heap) (dictionary address : Nat)
    (env : HeapCalls.Env) (globals : Mutable.Globals) (view : View heap dictionary buckets)
    (partition : StringValueBuckets.Partition solve.keyOfValue (solve.groupValues words) buckets)
    (found : heap[address]? = some (.list ((groups.selected buckets).map groups.reference)))
    (getGroups : env "anagram_groups" = some (.ref address)) :
    Evaluates program env expression heap (.ref heap.length) (work heap (collect [] (widths words (groups.selected buckets)))) globals := by
  let inner := comprehensionTargetsEnv env [(.name "g", .name "anagram_groups", []), (.name "w", .name "g", [])]
  have rows (bucket : Bucket) (member : bucket ∈ groups.selected buckets) :
      heap[bucket.address]? = some (.list ((wordsIn words bucket).map solve.text)) := by
    have inside := (List.mem_filter.mp member).1
    rw [← bucket_values words buckets partition bucket inside]
    exact view.lists bucket inside
  obtain ⟨finalEnv, executed⟩ := group_loop words heap address [] (groups.selected buckets) [] inner globals (by simpa using found) rows
  apply Evaluates.set_comp_sequence (finalEnv := finalEnv) (items := (groups.selected buckets).map groups.reference)
    (by intros; simp) (by intros; simp) (by intros; simp) (Evaluates.lookup getGroups)
  · simp [Mutable.sequence, found]
  · exact executed

/-- Sorting the scalar set produces the precise increasing list of needed widths in fresh heap cells. -/
theorem result (words : List String) (buckets : List Bucket) (heap : Mutable.Heap) (dictionary address : Nat)
    (env : HeapCalls.Env) (globals : Mutable.Globals) (view : View heap dictionary buckets)
    (partition : StringValueBuckets.Partition solve.keyOfValue (solve.groupValues words) buckets)
    (found : heap[address]? = some (.list ((groups.selected buckets).map groups.reference)))
    (getGroups : env "anagram_groups" = some (.ref address)) :
    Evaluates program env sortedExpression heap (.ref (heap.length + 1))
      (work heap (collect [] (widths words (groups.selected buckets))) ++ [.list ((needed words buckets).map number)]) globals := by
  apply Evaluates.builtin (.cons (set_result words buckets heap dictionary address env globals view partition found getGroups) .nil)
  simp [primitive, work, HeapSort.sort_scalars, ScalarSort.sort_ints, Mutable.allocate, Except.map,
    needed, List.map_map, Function.comp_def]

theorem mem_needed (words : List String) (buckets : List Bucket) (width : Int) :
    width ∈ needed words buckets ↔ ∃ bucket ∈ groups.selected buckets, ∃ word ∈ wordsIn words bucket, (word.length : Int) = width := by
  rw [needed, (ScalarSort.ints_perm _).mem_iff, mem_collect]
  simp [widths]

theorem positive (words : List String) (buckets : List Bucket) (width : Int) (member : width ∈ needed words buckets) : 0 < width := by
  obtain ⟨bucket, _, word, contains, rfl⟩ := (mem_needed words buckets width).mp member
  have nonempty : word ≠ "" := by simpa [wordsIn] using (List.mem_filter.mp (List.mem_filter.mp contains).1).2
  have positive : 0 < word.length := by
    have chars : word.toList ≠ [] := by simpa using nonempty
    exact List.length_pos_iff.mpr chars
  omega

end solvers.«98».lengths
