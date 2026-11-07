import solvers.«98.py.lean.d».SquareMembership

namespace solvers.«98».groups
open ProjectEulerSolutions
open StringValueBuckets (Bucket)
set_option Elab.async false

theorem word_member (words : List String) (bucket : Bucket) (word : String)
    (member : word ∈ lengths.wordsIn words bucket) :
    word ∈ words ∧ word ≠ "" ∧ AnagramKeys.key word = bucket.key := by
  simpa [lengths.wordsIn, and_assoc, and_left_comm, and_comm] using member

theorem word_nonempty (words : List String) (bucket : Bucket) (word : String)
    (member : word ∈ lengths.wordsIn words bucket) : word.toList ≠ [] := by
  have different := (word_member words bucket word member).2.1
  intro empty
  apply different
  exact String.toList_injective empty

theorem same_characters (words : List String) (bucket : Bucket) (left right : String)
    (first : left ∈ lengths.wordsIn words bucket) (second : right ∈ lengths.wordsIn words bucket) :
    left.toList.Perm right.toList :=
  (AnagramKeys.key_eq_iff_permutation left right).mp
    ((word_member words bucket left first).2.2.trans (word_member words bucket right second).2.2.symm)

theorem same_width (words : List String) (bucket : Bucket) (left right : String)
    (first : left ∈ lengths.wordsIn words bucket) (second : right ∈ lengths.wordsIn words bucket) : left.length = right.length := by
  simpa using (same_characters words bucket left right first second).length_eq

theorem selected_size (words : List String) (buckets : List Bucket)
    (partition : StringValueBuckets.Partition solve.keyOfValue (solve.groupValues words) buckets)
    (bucket : Bucket) (member : bucket ∈ selected buckets) : 2 ≤ (lengths.wordsIn words bucket).length := by
  have inside := (List.mem_filter.mp member).1
  have size : 2 ≤ bucket.values.length := of_decide_eq_true (List.mem_filter.mp member).2
  simpa only [lengths.bucket_values words buckets partition bucket inside, List.length_map] using size

/-- Every retained partner is nonempty and all of its letters are present in the current word's mapping. -/
theorem partners_valid (words : List String) (bucket : Bucket) (word : String)
    (member : word ∈ lengths.wordsIn words bucket) :
    ∀ other ∈ others.values (lengths.wordsIn words bucket) word,
      other.toList ≠ [] ∧ ∀ letter ∈ other.toList, letter ∈ word.toList := by
  intro other present
  have inside := (List.mem_filter.mp present).1
  exact ⟨word_nonempty words bucket other inside,
    fun letter occurs => (same_characters words bucket other word inside member).mem_iff.mp occurs⟩

end solvers.«98».groups
