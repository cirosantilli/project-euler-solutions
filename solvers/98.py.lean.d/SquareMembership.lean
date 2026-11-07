import solvers.«98.py.lean.d».CandidateLoop
import ProjectEulerSolutions.Lib.DecimalCharacters

namespace solvers.«98».mapped_word
open ProjectEulerSolutions ProjectEulerSolutions.AnagramSquares
open AnagramSearch (squareText)
open squares_with_length (squares)
set_option Elab.async false

theorem integer_square (root : Nat) : integerValue (squareText root) = root * root := by
  simp [integerValue, squareText, Nat.toList_repr]

/-- For canonical decimal strings of the cached width, the Python set test is exactly the reference square test. -/
theorem member_iff_square (width : Nat) (text : String) (first : Char) (rest : List Char)
    (view : text.toList = first :: rest) (decimal : ∀ digit ∈ text.toList, digit.isDigit)
    (nonzero : first ≠ '0') (lengths : text.length = width) :
    text ∈ squares width ↔ (integerValue text).sqrt ^ 2 = integerValue text := by
  constructor
  · intro member
    obtain ⟨root, _, _, rfl⟩ := (squares_with_length.mem_squares width text).mp member
    simp [integer_square, Nat.pow_two]
  · intro square
    have multiply : (integerValue text).sqrt * (integerValue text).sqrt = integerValue text := by
      simpa only [Nat.pow_two] using square
    have count : width = rest.length + 1 := lengths.symm.trans (by simpa using congrArg List.length view)
    have valueEq : DecimalCharacters.value (first :: rest) = integerValue text := by
      simp [DecimalCharacters.value, integerValue, view]
    have lower := DecimalCharacters.lower first rest (decimal first (by simp [view])) nonzero
    have upper := DecimalCharacters.upper (first :: rest) (by simpa only [← view] using decimal)
    have printed : toString (integerValue text) = text := by
      apply String.toList_injective
      rw [view, ← valueEq]
      exact DecimalCharacters.canonical first rest (by simpa only [← view] using decimal) nonzero
    apply (squares_with_length.mem_squares width text).mpr
    refine ⟨(integerValue text).sqrt, ?_, ?_, ?_⟩
    · simpa only [multiply, count, Nat.add_sub_cancel, valueEq] using lower
    · simpa only [multiply, count, List.length_cons, valueEq] using upper
    · simpa only [squareText, multiply] using printed

/-- A matched candidate's partner score is identical to the exhaustive reference assignment score. -/
theorem score_eq_reference (word other : String) (root : Nat) (best : Int)
    (matched : labels [] word.toList = labels [] (squareText root).toList)
    (included : ∀ letter ∈ other.toList, letter ∈ word.toList) (lengths : other.length = word.length) :
    score word.length word.toList (squareText root).toList other (integerValue (squareText root)) best =
      AnagramSearch.step word other root best := by
  cases view : other.toList with
  | nil => simp [score, AnagramSearch.step, view]
  | cons first rest =>
      by_cases zero : substitution '0' word.toList (squareText root).toList first = '0'
      · simp [score, AnagramSearch.step, view, matched, zero]
      · let text := String.ofList (other.toList.map (substitution '0' word.toList (squareText root).toList))
        have mapped := substitution_map '0' word.toList (squareText root).toList
          (compatible_of_labels_eq _ _ matched)
        have decimal : ∀ digit ∈ text.toList, digit.isDigit := by
          intro digit member
          obtain ⟨letter, present, rfl⟩ := List.mem_map.mp (by simpa [text] using member)
          apply AnagramSearch.square_decimal root
          have occurs := List.mem_map_of_mem (f := substitution '0' word.toList (squareText root).toList) (included letter present)
          simpa only [mapped] using occurs
        have member := member_iff_square word.length text
          (substitution '0' word.toList (squareText root).toList first)
          (rest.map (substitution '0' word.toList (squareText root).toList))
          (by simp [text, view]) decimal zero (by simpa [text] using lengths)
        have converted : integerValue text = AnagramSearch.transformed word.toList (squareText root).toList other.toList := by
          simp [integerValue, text, AnagramSearch.transformed]
        simp only [score, AnagramSearch.step, view, if_pos matched, if_neg zero]
        rw [← show text = String.ofList ((first :: rest).map (substitution '0' word.toList (squareText root).toList)) by simp [text, view]]
        simp only [member, converted, integer_square, AnagramSearch.transformed, view, Int.natCast_mul]

end solvers.«98».mapped_word
