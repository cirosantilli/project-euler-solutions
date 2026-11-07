import solvers.«98.py.lean.d».Execution
import ProjectEulerSolutions.Lib.AnagramReverse

namespace solvers.«98».maximum
open ProjectEulerSolutions ProjectEulerSolutions.AnagramSquares ProjectEulerSolutions.AnagramMaximum
open AnagramSearch (squareText)
set_option Elab.async false

theorem partner_bound (width : Nat) (word digits : List Char) (other : String) (firstSquare best bound : Int) :
    mapped_word.score width word digits other firstSquare best ≤ bound ↔
      best ≤ bound ∧ mapped_word.score width word digits other firstSquare bound ≤ bound := by
  cases view : other.toList with
  | nil => simp [mapped_word.score, view]
  | cons first rest =>
      by_cases zero : substitution '0' word digits first = '0'
      · simp [mapped_word.score, view, zero]
      · by_cases member : String.ofList (other.toList.map (substitution '0' word digits)) ∈ squares_with_length.squares width
        · simp_all [mapped_word.score]
        · simp_all [mapped_word.score]

theorem partners_bound (width : Nat) (word digits : List Char) (others : List String) (firstSquare best bound : Int) :
    mapped_word.scan width word digits others firstSquare best ≤ bound ↔
      best ≤ bound ∧ ∀ other ∈ others, mapped_word.score width word digits other firstSquare bound ≤ bound :=
  fold_bound _ _ bound (fun current other => partner_bound width word digits other firstSquare current bound) others best

theorem squares_bound (width : Nat) (word : String) (others squares : List String) (best bound : Int) :
    square_search.scan width word others squares best ≤ bound ↔
      best ≤ bound ∧ ∀ square ∈ squares, ∀ other ∈ others,
        mapped_word.score width word.toList square.toList other (mapped_word.integerValue square) bound ≤ bound :=
  fold_bound _ _ bound (fun current square => partners_bound width word.toList square.toList others (mapped_word.integerValue square) current bound) squares best

theorem raw_word_bound (width : Nat) (group : List String) (word : String) (best bound : Int) :
    word_search.score width group word best ≤ bound ↔
      best ≤ bound ∧ ∀ square ∈ candidates.values width word, ∀ other ∈ others.values group word,
        mapped_word.score width word.toList square.toList other (mapped_word.integerValue square) bound ≤ bound :=
  squares_bound _ _ _ _ _ _

theorem words_bound (width : Nat) (group remaining : List String) (best bound : Int) :
    word_search.scan width group remaining best ≤ bound ↔
      best ≤ bound ∧ ∀ word ∈ remaining, word_search.score width group word bound ≤ bound := by
  apply fold_bound
  intro current word
  simp only [raw_word_bound, le_refl, true_and]

theorem groups_bound (words : List String) (buckets : List StringValueBuckets.Bucket) (best bound : Int) :
    group_search.scan words buckets best ≤ bound ↔
      best ≤ bound ∧ ∀ bucket ∈ buckets, group_search.score words bucket bound ≤ bound := by
  apply fold_bound
  intro current bucket
  simp only [group_search.score, words_bound, le_refl, true_and]

/-- Pattern filtering and the cached square test retain exactly the reference assignments for each partner. -/
theorem word_bound (group : List String) (word : String) (best bound : Int)
    (valid : ∀ other ∈ others.values group word, other.length = word.length ∧ ∀ letter ∈ other.toList, letter ∈ word.toList) :
    word_search.score word.length group word best ≤ bound ↔
      best ≤ bound ∧ ∀ other ∈ others.values group word, PairBound word other bound := by
  rw [raw_word_bound]
  constructor
  · rintro ⟨bestBound, checked⟩
    refine ⟨bestBound, ?_⟩
    intro other present root lower upper accepted
    have matched := (accepted_conditions word other root accepted).1
    have limits := (root_interval word.length root).mp ⟨lower, upper⟩
    have cached : squareText root ∈ candidates.values word.length word := by
      apply List.mem_filter.mpr
      refine ⟨(squares_with_length.mem_squares _ _).mpr ⟨root, limits.1, limits.2, rfl⟩, ?_⟩
      simpa only [squares_with_length.signature, decide_eq_true_eq] using matched.symm
    have score := checked (squareText root) cached other present
    rw [mapped_word.score_eq_reference word other root bound matched (valid other present).2 (valid other present).1] at score
    exact ((step_bound word other root bound bound).mp score).2 accepted
  · rintro ⟨bestBound, bounded⟩
    refine ⟨bestBound, ?_⟩
    intro square cached other present
    have matched := candidates.matched word.length word square cached
    obtain ⟨root, lower, upper, rfl⟩ := (squares_with_length.mem_squares _ _).mp (List.mem_filter.mp cached).1
    have limits := (root_interval word.length root).mpr ⟨lower, upper⟩
    rw [mapped_word.score_eq_reference word other root bound matched (valid other present).2 (valid other present).1]
    exact (step_bound word other root bound bound).mpr ⟨le_refl _, bounded other present root limits.1 limits.2⟩

end solvers.«98».maximum
