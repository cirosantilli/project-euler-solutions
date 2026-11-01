import ProjectEulerSolutions.Lib.PrimePowerTriples
import Mathlib.Data.Finset.Dedup

/-! List enumeration and sorted-prefix pruning preserve exactly the distinct prime-power sums. -/
namespace ProjectEulerSolutions.PrimePowerTriples
set_option Elab.async false

def referenceValues (limit : Nat) (primes : List Nat) : List Nat :=
  primes.flatMap fun p => primes.flatMap fun q =>
    (primes.map (fun r => p ^ 2 + q ^ 3 + r ^ 4)).filter (fun value => decide (value < limit))

theorem mem_referenceValues (limit value : Nat) (primes : List Nat) :
    value ∈ referenceValues limit primes ↔ value < limit ∧
      ∃ p ∈ primes, ∃ q ∈ primes, ∃ r ∈ primes, value = p ^ 2 + q ^ 3 + r ^ 4 := by
  simp only [referenceValues, List.mem_flatMap, List.mem_filter, List.mem_map, decide_eq_true_eq]
  constructor
  · rintro ⟨p, hp, q, hq, ⟨r, hr, equal⟩, inside⟩
    exact ⟨inside, p, hp, q, hq, r, hr, equal.symm⟩
  · rintro ⟨inside, p, hp, q, hq, r, hr, equal⟩
    exact ⟨p, hp, q, hq, ⟨r, hr, equal.symm⟩, inside⟩

theorem referenceValues_complete (limit : Nat) (primes : List Nat)
    (complete : ∀ p, p ∈ primes ↔ p ∈ primesThrough (Nat.sqrt limit)) :
    (referenceValues limit primes).toFinset = exhaustive limit := by
  ext value
  simp only [List.mem_toFinset, mem_referenceValues, complete,
    exhaustive, Finset.mem_biUnion, Finset.mem_filter, Finset.mem_image]
  constructor
  · rintro ⟨inside, p, hp, q, hq, r, hr, equal⟩
    exact ⟨p, hp, q, hq, ⟨r, hr, equal.symm⟩, inside⟩
  · rintro ⟨p, hp, q, hq, ⟨r, hr, equal⟩, inside⟩
    exact ⟨inside, p, hp, q, hq, r, hr, equal.symm⟩

def rowValues (limit offset : Nat) (squares : List Nat) : List Nat :=
  (squares.takeWhile (fun s => decide (offset + s < limit))).map (offset + ·)

theorem mem_rowValues (limit offset value : Nat) (squares : List Nat)
    (ordered : squares.Pairwise (· ≤ ·)) :
    value ∈ rowValues limit offset squares ↔ value < limit ∧ ∃ s ∈ squares, value = offset + s := by
  rw [rowValues, sorted_prefix limit offset squares ordered]
  simp only [List.mem_map, List.mem_filter, decide_eq_true_eq]
  constructor
  · rintro ⟨s, ⟨hs, inside⟩, rfl⟩
    exact ⟨inside, s, hs, rfl⟩
  · rintro ⟨inside, s, hs, rfl⟩
    exact ⟨s, ⟨hs, inside⟩, rfl⟩

def prunedValues (limit : Nat) (squares cubes fourths : List Nat) : List Nat :=
  fourths.flatMap fun f =>
    (cubes.takeWhile (fun c => decide (f + c < limit))).flatMap fun c =>
      rowValues limit (f + c) squares

theorem mem_prunedValues (limit value : Nat) (squares cubes fourths : List Nat)
    (orderedSquares : squares.Pairwise (· ≤ ·)) (orderedCubes : cubes.Pairwise (· ≤ ·)) :
    value ∈ prunedValues limit squares cubes fourths ↔ value < limit ∧
      ∃ f ∈ fourths, ∃ c ∈ cubes, ∃ s ∈ squares, value = f + c + s := by
  simp only [prunedValues, List.mem_flatMap, sorted_prefix limit _ cubes orderedCubes,
    List.mem_filter, decide_eq_true_eq, mem_rowValues limit _ value squares orderedSquares]
  constructor
  · rintro ⟨f, hf, c, ⟨hc, _⟩, inside, s, hs, equal⟩
    exact ⟨inside, f, hf, c, hc, s, hs, equal⟩
  · rintro ⟨inside, f, hf, c, hc, s, hs, equal⟩
    exact ⟨f, hf, c, ⟨hc, by omega⟩, inside, s, hs, equal⟩

/-- The two source breaks preserve the same finite set as exhaustive triple enumeration. -/
theorem prunedValues_complete (limit : Nat) (squares cubes fourths : List Nat)
    (orderedSquares : squares.Pairwise (· ≤ ·)) (orderedCubes : cubes.Pairwise (· ≤ ·))
    (squareValues : squares.toFinset = powers limit 2)
    (cubeValues : cubes.toFinset = powers limit 3)
    (fourthValues : fourths.toFinset = powers limit 4) :
    (prunedValues limit squares cubes fourths).toFinset = exhaustive limit := by
  rw [grouping_equivalence]
  ext value
  have squareMember (s : Nat) : s ∈ squares ↔ s ∈ powers limit 2 := by
    rw [← squareValues, List.mem_toFinset]
  have cubeMember (c : Nat) : c ∈ cubes ↔ c ∈ powers limit 3 := by
    rw [← cubeValues, List.mem_toFinset]
  have fourthMember (f : Nat) : f ∈ fourths ↔ f ∈ powers limit 4 := by
    rw [← fourthValues, List.mem_toFinset]
  simp only [List.mem_toFinset, mem_prunedValues limit value squares cubes fourths orderedSquares orderedCubes,
    squareMember, cubeMember, fourthMember, grouped, Finset.mem_biUnion, Finset.mem_filter, Finset.mem_image]
  constructor
  · rintro ⟨inside, f, hf, c, hc, s, hs, equal⟩
    exact ⟨f, hf, c, hc, ⟨s, hs, equal.symm⟩, inside⟩
  · rintro ⟨f, hf, c, hc, ⟨s, hs, equal⟩, inside⟩
    exact ⟨inside, f, hf, c, hc, s, hs, equal.symm⟩

end ProjectEulerSolutions.PrimePowerTriples
