import solvers.«98.py.lean.d».Squares

namespace solvers.«98».squares_with_length
open FormalSW.PyAst ProjectEulerSolutions
open AnagramSquares (firstRoot lastRoot)
open AnagramSearch (squareText)
set_option Elab.async false

private theorem contains_insert_string (items : List Constant) (value query : String) :
    ScalarSet.contains (ScalarSet.insert items (.string value)) (.string query) =
      (ScalarSet.contains items (.string query) || decide (value = query)) := by
  by_cases same : value = query
  · subst query
    simp [ScalarSet.contains_insert_self]
  · unfold ScalarSet.insert
    split
    · simp [same]
    · simp [ScalarSet.contains, scalarCompare, same]

private theorem contains_fold (items : List Constant) (strings : List String) (query : String) :
    ScalarSet.contains ((strings.map Constant.string).foldl ScalarSet.insert items) (.string query) =
      (ScalarSet.contains items (.string query) || decide (query ∈ strings)) := by
  induction strings generalizing items with
  | nil => simp
  | cons first rest ih =>
      simp [ih, contains_insert_string, Bool.or_assoc, eq_comm]

/-- Membership in the actual Python set is independent of its retained representative order. -/
theorem contains_setValues (strings : List String) (query : String) :
    ScalarSet.contains (setValues strings) (.string query) = decide (query ∈ strings) := by
  simpa [setValues, ScalarSet.contains] using contains_fold [] strings query

/-- The root interval in the source enumerates exactly the squares with this decimal width. -/
theorem mem_squares (width : Nat) (text : String) : text ∈ squares width ↔
    ∃ root : Nat, 10 ^ (width - 1) ≤ root * root ∧ root * root < 10 ^ width ∧ squareText root = text := by
  have ordered := AnagramSearch.ordered_roots width
  have stop : firstRoot width + (lastRoot width + 1 - firstRoot width) = lastRoot width + 1 := by omega
  simp only [squares, rangeSquares, List.mem_map, List.mem_range'_1, stop]
  constructor
  · rintro ⟨root, ⟨lower, upper⟩, printed⟩
    obtain ⟨lowSquare, highSquare⟩ := (AnagramSquares.root_interval width root).mp ⟨lower, by omega⟩
    exact ⟨root, lowSquare, highSquare, printed⟩
  · rintro ⟨root, lower, upper, printed⟩
    obtain ⟨lowRoot, highRoot⟩ := (AnagramSquares.root_interval width root).mpr ⟨lower, upper⟩
    exact ⟨root, ⟨lowRoot, by omega⟩, printed⟩

theorem width_of_member (width : Nat) (positive : 0 < width) (text : String) (member : text ∈ squares width) :
    text.length = width := by
  obtain ⟨root, lower, upper, rfl⟩ := (mem_squares width text).mp member
  obtain ⟨lowRoot, highRoot⟩ := (AnagramSquares.root_interval width root).mpr ⟨lower, upper⟩
  exact AnagramSearch.square_width width root positive lowRoot highRoot

end solvers.«98».squares_with_length
