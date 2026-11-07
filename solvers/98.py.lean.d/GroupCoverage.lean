import solvers.«98.py.lean.d».GroupFilter

namespace solvers.«98».groups
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
open StringValueBuckets (Bucket)
open pattern_signature (program number)
set_option Elab.async false

/-- Filtering out singleton buckets loses no distinct anagram pair, including with duplicate input words. -/
theorem pair_covered (words : List String) (buckets : List Bucket)
    (partition : StringValueBuckets.Partition solve.keyOfValue (solve.groupValues words) buckets)
    (left right : String) (pair : (left, right) ∈ AnagramPairs.pairs words) :
    ∃ bucket ∈ selected buckets, solve.text left ∈ bucket.values ∧ solve.text right ∈ bucket.values := by
  obtain ⟨leftMember, rightMember, different, equal⟩ := AnagramPairs.mem_pairs_sound words left right pair
  obtain ⟨leftChars, rightChars⟩ := AnagramPairs.mem_pairs_nonempty words left right pair
  have leftNonempty : left ≠ "" := by intro empty; simp [empty] at leftChars
  have rightNonempty : right ≠ "" := by intro empty; simp [empty] at rightChars
  have first : solve.text left ∈ solve.groupValues words := by
    apply List.mem_map.mpr
    exact ⟨left, List.mem_filter.mpr ⟨leftMember, by simpa using leftNonempty⟩, rfl⟩
  have second : solve.text right ∈ solve.groupValues words := by
    apply List.mem_map.mpr
    exact ⟨right, List.mem_filter.mpr ⟨rightMember, by simpa using rightNonempty⟩, rfl⟩
  obtain ⟨bucket, member, keyEqual, containsLeft⟩ := partition.member (solve.text left) first
  have containsRight : solve.text right ∈ bucket.values := by
    rw [partition.bucket_values bucket member]
    apply List.mem_filter.mpr
    refine ⟨second, ?_⟩
    simp only [decide_eq_true_eq, keyEqual]
    exact equal.symm
  have distinct : solve.text left ≠ solve.text right := by
    intro same
    apply different
    simpa only [solve.text, Mutable.Value.scalar.injEq, Constant.string.injEq] using same
  have length : 2 ≤ bucket.values.length := by
    cases rows : bucket.values with
    | nil => simp [rows] at containsLeft
    | cons first rest =>
        cases rest with
        | nil =>
            simp only [rows, List.mem_singleton] at containsLeft containsRight
            exact (distinct (containsLeft.trans containsRight.symm)).elim
        | cons second tail => simp
  exact ⟨bucket, List.mem_filter.mpr ⟨member, by simpa using length⟩, containsLeft, containsRight⟩

theorem no_pairs (words : List String) (buckets : List Bucket)
    (partition : StringValueBuckets.Partition solve.keyOfValue (solve.groupValues words) buckets)
    (empty : selected buckets = []) : AnagramPairs.pairs words = [] := by
  apply List.eq_nil_iff_forall_not_mem.mpr
  intro pair present
  obtain ⟨bucket, member, _, _⟩ := pair_covered words buckets partition pair.1 pair.2 present
  simp [empty] at member

abbrev emptyTest : Expr := .unaryOp .not (.name "anagram_groups")
abbrev emptyGuard : Stmt := .ifThen emptyTest [.return (.constant (.int 0))] []

theorem source_guard : (((findFunctionIn program.body "solve").getD ([], [])).2)[3]? = some emptyGuard := rfl

theorem empty_test (buckets : List Bucket) (heap : Mutable.Heap) (address : Nat) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (found : heap[address]? = some (.list ((selected buckets).map reference)))
    (getGroups : env "anagram_groups" = some (.ref address)) :
    Evaluates program env emptyTest heap (.scalar (.bool (decide (selected buckets = [])))) heap globals := by
  apply Evaluates.unary (Evaluates.lookup getGroups)
  simp [HeapCalls.unary, Mutable.truth, found]
  cases selected buckets <;> rfl

theorem empty_branch (buckets : List Bucket) (heap : Mutable.Heap) (address : Nat) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (empty : selected buckets = []) (found : heap[address]? = some (.list ((selected buckets).map reference)))
    (getGroups : env "anagram_groups" = some (.ref address)) (following : List Stmt) :
    BlockReturns program env (emptyGuard :: following) heap (number 0) heap globals := by
  apply BlockReturns.branch (flag := true) (empty_test buckets heap address env globals found getGroups)
    (by simp [Mutable.truth, scalarTruth, empty])
  exact BlockReturns.return_value Evaluates.constant

theorem nonempty_guard (buckets : List Bucket) (heap : Mutable.Heap) (address : Nat) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (nonempty : selected buckets ≠ []) (found : heap[address]? = some (.list ((selected buckets).map reference)))
    (getGroups : env "anagram_groups" = some (.ref address)) :
    BlockContinues program env [emptyGuard] heap env heap globals := by
  apply BlockContinues.branch (flag := false) (empty_test buckets heap address env globals found getGroups)
    (by simp [Mutable.truth, scalarTruth, nonempty])
  exact BlockContinues.nil

end solvers.«98».groups
