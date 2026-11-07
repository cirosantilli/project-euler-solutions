import ProjectEulerSolutions.Lib.AnagramMaximum
import ProjectEulerSolutions.Lib.DecimalCharacters

/-! Reversing the orientation of a pair preserves its valid square assignments. -/
namespace ProjectEulerSolutions.AnagramMaximum
open AnagramSquares AnagramSearch
set_option Elab.async false

theorem substitution_of_map (word : List Char) (f : Char → Char) (letter : Char) (member : letter ∈ word) :
    substitution '0' word (word.map f) letter = f letter := by
  simp [substitution, List.getElem?_map, List.getElem?_idxOf member]

theorem mapped_decimal (word other : String) (root : Nat)
    (matched : labels [] word.toList = labels [] (squareText root).toList)
    (included : ∀ letter ∈ other.toList, letter ∈ word.toList) :
    ∀ digit ∈ other.toList.map (substitution '0' word.toList (squareText root).toList), digit.isDigit := by
  have mapped := substitution_map '0' word.toList (squareText root).toList (compatible_of_labels_eq _ _ matched)
  intro digit member
  obtain ⟨letter, present, rfl⟩ := List.mem_map.mp member
  apply square_decimal root
  have occurs := List.mem_map_of_mem (f := substitution '0' word.toList (squareText root).toList) (included letter present)
  simpa only [mapped] using occurs

theorem square_first_nonzero (width root : Nat) (positive : 0 < width)
    (lower : firstRoot width ≤ root) (upper : root ≤ lastRoot width)
    (first : Char) (rest : List Char) (view : (squareText root).toList = first :: rest) : first ≠ '0' := by
  intro zero
  have count : rest.length + 1 = width := by simpa only [view, List.length_cons] using square_width width root positive lower upper
  have low := ((root_interval width root).mp ⟨lower, upper⟩).1
  have restDigits : ∀ digit ∈ rest, digit.isDigit := by
    intro digit member
    exact square_decimal root digit (by simp [view, member])
  have restBound := DecimalCharacters.upper rest restDigits
  have value : DecimalCharacters.value (first :: rest) = root * root := by
    rw [← view]
    simp [DecimalCharacters.value, squareText, Nat.toList_repr]
  rw [DecimalCharacters.cons_value, zero] at value
  simp only [Char.reduceToNat, Nat.sub_self, Nat.mul_zero, Nat.zero_add] at value
  rw [← count, Nat.add_sub_cancel] at low
  omega

/-- The transformed partner square prints exactly its assigned digits and lies in the same width interval. -/
theorem transformed_square (word other : String) (root : Nat)
    (included : ∀ letter ∈ other.toList, letter ∈ word.toList) (accepted : Accepted word other root) :
    let n := transformed word.toList (squareText root).toList other.toList
    firstRoot other.length ≤ n.sqrt ∧ n.sqrt ≤ lastRoot other.length ∧
      squareText n.sqrt = String.ofList (other.toList.map (substitution '0' word.toList (squareText root).toList)) := by
  cases view : other.toList with
  | nil => simp [Accepted, view] at accepted
  | cons first rest =>
      obtain ⟨matched, nonzero, square⟩ := (show labels [] word.toList = labels [] (squareText root).toList ∧
          substitution '0' word.toList (squareText root).toList first ≠ '0' ∧
          (transformed word.toList (squareText root).toList other.toList).sqrt ^ 2 =
            transformed word.toList (squareText root).toList other.toList from by simpa [Accepted, view] using accepted)
      let f := substitution '0' word.toList (squareText root).toList
      let n := transformed word.toList (squareText root).toList other.toList
      have digits : ∀ digit ∈ f first :: rest.map f, digit.isDigit := by
        simpa only [view, List.map_cons] using mapped_decimal word other root matched included
      have value : DecimalCharacters.value (f first :: rest.map f) = n := by simp [DecimalCharacters.value, n, transformed, view, f]
      have count : other.length = rest.length + 1 := by simpa using congrArg List.length view
      have multiply : n.sqrt * n.sqrt = n := by simpa only [Nat.pow_two] using square
      have low : 10 ^ (other.length - 1) ≤ n := by
        simpa only [count, Nat.add_sub_cancel, List.length_map, value] using
          DecimalCharacters.lower (f first) (rest.map f) (digits (f first) (by simp)) nonzero
      have high : n < 10 ^ other.length := by
        simpa only [count, List.length_cons, List.length_map, value] using DecimalCharacters.upper (f first :: rest.map f) digits
      have bounds := (root_interval other.length n.sqrt).mpr ⟨by simpa only [multiply] using low, by simpa only [multiply] using high⟩
      rw [← view]
      refine ⟨bounds.1, bounds.2, ?_⟩
      change squareText n.sqrt = _
      rw [squareText, multiply]
      apply String.toList_injective
      rw [String.toList_ofList, view, List.map_cons, ← value]
      exact DecimalCharacters.canonical (f first) (rest.map f) digits nonzero

/-- Every accepted assignment has a reverse assignment whose second value is the original first square. -/
theorem accepted_reverse (word other : String) (root : Nat) (permutation : word.toList.Perm other.toList)
    (lower : firstRoot word.length ≤ root) (upper : root ≤ lastRoot word.length)
    (accepted : Accepted word other root) :
    let n := transformed word.toList (squareText root).toList other.toList
    firstRoot other.length ≤ n.sqrt ∧ n.sqrt ≤ lastRoot other.length ∧ Accepted other word n.sqrt ∧
      transformed other.toList (squareText n.sqrt).toList word.toList = root * root := by
  have matched : labels [] word.toList = labels [] (squareText root).toList := by
    cases view : other.toList with
    | nil => simp [Accepted, view] at accepted
    | cons first rest => exact (show _ ∧ _ ∧ _ from by simpa [Accepted, view] using accepted).1
  let f := substitution '0' word.toList (squareText root).toList
  let n := transformed word.toList (squareText root).toList other.toList
  obtain ⟨low, high, printed⟩ := transformed_square word other root (fun letter member => permutation.mem_iff.mpr member) accepted
  change firstRoot other.length ≤ n.sqrt at low
  change n.sqrt ≤ lastRoot other.length at high
  change squareText n.sqrt = String.ofList (other.toList.map f) at printed
  have mapped : word.toList.map f = (squareText root).toList := substitution_map '0' _ _ (compatible_of_labels_eq _ _ matched)
  have injective := substitution_injective '0' word.toList (squareText root).toList (compatible_of_labels_eq _ _ matched)
  have reverseMatched : labels [] other.toList = labels [] (squareText n.sqrt).toList := by
    apply (labels_eq_iff_substitution '0' _ _).mpr
    refine ⟨f, by simp [printed], ?_⟩
    intro left leftMember right rightMember same
    exact injective (permutation.mem_iff.mpr leftMember) (permutation.mem_iff.mpr rightMember) same
  have reversed (letter : Char) (member : letter ∈ word.toList) :
      substitution '0' other.toList (squareText n.sqrt).toList letter = f letter := by
    rw [printed, String.toList_ofList]
    exact substitution_of_map other.toList f letter (permutation.mem_iff.mp member)
  have back : word.toList.map (substitution '0' other.toList (squareText n.sqrt).toList) = (squareText root).toList :=
    (List.map_congr_left reversed).trans mapped
  have backValue : transformed other.toList (squareText n.sqrt).toList word.toList = root * root := by
    simp only [transformed, back]
    simp [squareText, Nat.toList_repr]
  refine ⟨low, high, ?_, backValue⟩
  change Accepted other word n.sqrt
  have nonempty : word.toList ≠ [] := by
    intro empty
    have length := congrArg List.length matched
    have positive := Nat.length_repr_pos (n := root * root)
    simp only [labels_length, empty, List.length_nil] at length
    change 0 = (root * root).repr.length at length
    omega
  obtain ⟨first, rest, view⟩ := List.exists_cons_of_ne_nil nonempty
  have positive : 0 < word.length := by simpa using List.length_pos_iff.mpr nonempty
  have digitView : (squareText root).toList = f first :: rest.map f := by rw [← mapped, view]; rfl
  have nonzero := square_first_nonzero word.length root positive lower upper (f first) (rest.map f) digitView
  have firstReverse : substitution '0' other.toList (squareText n.sqrt).toList first ≠ '0' := by
    rw [reversed first (by simp [view])]
    exact nonzero
  have square : (transformed other.toList (squareText n.sqrt).toList word.toList).sqrt ^ 2 =
      transformed other.toList (squareText n.sqrt).toList word.toList := by simp [backValue, Nat.pow_two]
  simpa only [Accepted, view] using And.intro reverseMatched (And.intro firstReverse square)

theorem accepted_conditions (word other : String) (root : Nat) (accepted : Accepted word other root) :
    labels [] word.toList = labels [] (squareText root).toList ∧
      (transformed word.toList (squareText root).toList other.toList).sqrt ^ 2 =
        transformed word.toList (squareText root).toList other.toList := by
  cases view : other.toList with
  | nil => simp [Accepted, view] at accepted
  | cons first rest =>
      have parts := (show _ ∧ _ ∧ _ from by simpa [Accepted, view] using accepted)
      exact ⟨parts.1, by simpa only [view] using parts.2.2⟩

theorem pair_bound_reverse (word other : String) (bound : Int) (permutation : word.toList.Perm other.toList)
    (bounded : PairBound other word bound) : PairBound word other bound := by
  intro root lower upper accepted
  let n := transformed word.toList (squareText root).toList other.toList
  obtain ⟨low, high, reverse, back⟩ := accepted_reverse word other root permutation lower upper accepted
  change transformed other.toList (squareText n.sqrt).toList word.toList = root * root at back
  have limits := bounded n.sqrt low high reverse
  have product : n.sqrt * n.sqrt = n := by simpa only [Nat.pow_two] using (accepted_conditions word other root accepted).2
  constructor
  · simpa only [back, Int.natCast_mul] using limits.2
  · have cast : (n.sqrt : Int) * n.sqrt = n := by exact_mod_cast product
    simpa only [cast] using limits.1

theorem pair_bound_symm (word other : String) (bound : Int) (permutation : word.toList.Perm other.toList) :
    PairBound word other bound ↔ PairBound other word bound :=
  ⟨pair_bound_reverse other word bound permutation.symm, pair_bound_reverse word other bound permutation⟩

end ProjectEulerSolutions.AnagramMaximum
