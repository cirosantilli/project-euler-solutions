import solvers.«98.py.lean.d».ScoreBounds

namespace solvers.«98».maximum
open FormalSW.PyAst ProjectEulerSolutions ProjectEulerSolutions.AnagramMaximum
open StringValueBuckets (Bucket)
set_option Elab.async false

theorem group_bound (words : List String) (bucket : Bucket) (best bound : Int) :
    group_search.score words bucket best ≤ bound ↔
      best ≤ bound ∧ ∀ word ∈ lengths.wordsIn words bucket,
        ∀ other ∈ others.values (lengths.wordsIn words bucket) word, PairBound word other bound := by
  rw [group_search.score, words_bound]
  have property (word : String) (member : word ∈ lengths.wordsIn words bucket) :
      word_search.score bucket.key.length (lengths.wordsIn words bucket) word bound ≤ bound ↔
        ∀ other ∈ others.values (lengths.wordsIn words bucket) word, PairBound word other bound := by
    rw [← group_search.word_width words bucket word member]
    have valid : ∀ other ∈ others.values (lengths.wordsIn words bucket) word,
        other.length = word.length ∧ ∀ letter ∈ other.toList, letter ∈ word.toList := by
      intro other present
      exact ⟨groups.same_width words bucket other word (List.mem_filter.mp present).1 member,
        (groups.partners_valid words bucket word member other present).2⟩
    simpa only [le_refl, true_and] using word_bound (lengths.wordsIn words bucket) word bound bound valid
  constructor
  · rintro ⟨initial, checked⟩
    exact ⟨initial, fun word member => (property word member).mp (checked word member)⟩
  · rintro ⟨initial, checked⟩
    exact ⟨initial, fun word member => (property word member).mpr (checked word member)⟩

/-- Grouping, repeated words, opposite pair orientations, and iteration order do not change the maximum's bounds. -/
theorem grouped_bound (words : List String) (buckets : List Bucket)
    (partition : StringValueBuckets.Partition solve.keyOfValue (solve.groupValues words) buckets) (best bound : Int) :
    group_search.scan words (groups.selected buckets) best ≤ bound ↔
      best ≤ bound ∧ ∀ pair ∈ AnagramPairs.pairs words, PairBound pair.1 pair.2 bound := by
  rw [groups_bound]
  constructor
  · rintro ⟨initial, checked⟩
    refine ⟨initial, ?_⟩
    rintro ⟨left, right⟩ member
    obtain ⟨bucket, selected, leftValues, rightValues⟩ := groups.pair_covered words buckets partition left right member
    have inside := (List.mem_filter.mp selected).1
    have contents := lengths.bucket_values words buckets partition bucket inside
    have leftIn : left ∈ lengths.wordsIn words bucket := by
      rw [contents] at leftValues
      simpa [solve.text] using leftValues
    have rightIn : right ∈ lengths.wordsIn words bucket := by
      rw [contents] at rightValues
      simpa [solve.text] using rightValues
    have different := (AnagramPairs.mem_pairs_sound words left right member).2.2.1
    exact ((group_bound words bucket bound bound).mp (checked bucket selected)).2 left leftIn right
      (List.mem_filter.mpr ⟨rightIn, by simpa using Ne.symm different⟩)
  · rintro ⟨initial, checked⟩
    refine ⟨initial, ?_⟩
    intro bucket selected
    apply (group_bound words bucket bound bound).mpr
    refine ⟨le_refl _, ?_⟩
    intro left leftIn right present
    obtain ⟨rightIn, different⟩ := List.mem_filter.mp present
    have notEqual : left ≠ right := Ne.symm (of_decide_eq_true different)
    have first := groups.word_member words bucket left leftIn
    have second := groups.word_member words bucket right rightIn
    rcases AnagramPairs.mem_pairs_complete words left right first.1 second.1 notEqual (first.2.2.trans second.2.2.symm) with forward | reverse
    · exact checked (left, right) forward
    · exact (pair_bound_symm left right bound (groups.same_characters words bucket left right leftIn rightIn)).mpr
        (checked (right, left) reverse)

/-- The grouped optimized search and the exhaustive reference search return the same integer. -/
theorem grouped_eq (words : List String) (buckets : List Bucket)
    (partition : StringValueBuckets.Partition solve.keyOfValue (solve.groupValues words) buckets) :
    group_search.scan words (groups.selected buckets) 0 = AnagramSearch.solve words := by
  apply le_antisymm
  · exact (grouped_bound words buckets partition 0 _).mpr ((solve_bound words _).mp (le_refl _))
  · exact (solve_bound words _).mpr ((grouped_bound words buckets partition 0 _).mp (le_refl _))

end solvers.«98».maximum
