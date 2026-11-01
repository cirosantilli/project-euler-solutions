import ProjectEulerSolutions.Lib.ProductSumSearch
import Mathlib.Data.List.Sort

/-! Soundness and completeness of the bounded, size-pruned product-sum search. -/
namespace ProjectEulerSolutions.ProductSumNumbers
set_option Elab.async false

theorem search_sound (maxSize limit start : Nat) (before : List Nat)
    (valid : Admissible before) (startBound : 2 ≤ start) (size value : Nat)
    (member : (size, value) ∈ search maxSize limit start before.prod before.sum before.length) :
    size ≤ maxSize ∧ value ≤ limit ∧ Represents size value := by
  have positive := product_positive before valid
  obtain ⟨factor, inside, accepted, here | later⟩ :=
    (mem_search maxSize limit start before.prod before.sum before.length size value
      (by omega) startBound).mp member
  · have bounds := (mem_nextFactors limit start before.prod factor (by omega)).mp inside
    have single : Admissible [factor] := by simp [Admissible]; omega
    have extended := represents_effective (before ++ [factor])
      (admissible_append before [factor] valid single)
    obtain ⟨sizeEq, valueEq⟩ := here
    refine ⟨by omega, by omega, ?_⟩
    have extendedSize : effectiveSize (before ++ [factor]) = size := by
      simpa [effectiveSize] using sizeEq.symm
    simpa only [extendedSize, List.prod_append, List.prod_cons, List.prod_nil, mul_one,
      ← valueEq] using extended
  · have bounds := (mem_nextFactors limit start before.prod factor (by omega)).mp inside
    have single : Admissible [factor] := by simp [Admissible]; omega
    exact search_sound maxSize limit factor (before ++ [factor])
      (admissible_append before [factor] valid single) (by omega) size value
      (by simpa using later)
termination_by limit + 1 - before.prod
decreasing_by
  simp only [List.prod_append, List.prod_cons, List.prod_nil, mul_one]
  exact descent limit before.prod factor positive (by omega) bounds.2

theorem search_complete (maxSize limit start : Nat) (before suffix : List Nat)
    (valid : Admissible before) (remaining : Admissible suffix)
    (ordered : suffix.Pairwise (· ≤ ·)) (lower : ∀ factor ∈ suffix, start ≤ factor)
    (startBound : 2 ≤ start) (nonempty : suffix ≠ [])
    (inside : (before ++ suffix).prod ≤ limit)
    (accepted : effectiveSize (before ++ suffix) ≤ maxSize) :
    (effectiveSize (before ++ suffix), (before ++ suffix).prod) ∈
      search maxSize limit start before.prod before.sum before.length := by
  induction suffix generalizing before start with
  | nil => contradiction
  | cons factor tail ih =>
      have factorBound := remaining factor (by simp)
      have factorLower := lower factor (by simp)
      have tailValid : Admissible tail := by
        intro x member
        exact remaining x (by simp [member])
      have single : Admissible [factor] := by simpa [Admissible] using factorBound
      have extendedValid := admissible_append before [factor] valid single
      have positive := product_positive before valid
      have tailPositive := product_positive tail tailValid
      have productBound : before.prod * factor ≤ limit := by
        simp only [List.prod_append, List.prod_cons] at inside
        nlinarith
      have firstSize := effectiveSize_prefix (before ++ [factor]) tail extendedValid tailValid
      have firstAccepted : before.length + 1 + before.prod * factor - (before.sum + factor) ≤ maxSize := by
        simp only [List.append_assoc, List.singleton_append] at firstSize
        have result := firstSize.trans accepted
        simpa [effectiveSize] using result
      apply (mem_search maxSize limit start before.prod before.sum before.length _ _
        (by omega) startBound).mpr
      refine ⟨factor, (mem_nextFactors limit start before.prod factor (by omega)).mpr
        ⟨factorLower, productBound⟩, firstAccepted, ?_⟩
      by_cases empty : tail = []
      · left
        simp [empty, effectiveSize]
      · right
        have tailOrdered := (List.pairwise_cons.mp ordered).2
        have tailLower := (List.pairwise_cons.mp ordered).1
        have result := ih factor (before ++ [factor]) extendedValid tailValid tailOrdered
          tailLower factorBound empty (by simpa using inside) (by simpa using accepted)
        simpa using result

/-- Sorting the non-one factors makes every representation reachable by the ordered DFS. -/
theorem root_complete (maxSize limit size value : Nat) (sizeBound : size ≤ maxSize)
    (nontrivial : 2 ≤ size) (valueBound : value ≤ limit) (represented : Represents size value) :
    (size, value) ∈ search maxSize limit 2 1 0 0 := by
  obtain ⟨factors, valid, productEq, sizeEq⟩ := (represents_iff_factors size value).mp represented
  let sorted := factors.mergeSort (fun a b => decide (a ≤ b))
  have permutation : sorted.Perm factors := List.mergeSort_perm _ _
  have sortedValid : Admissible sorted := by
    intro factor member
    exact valid factor (permutation.mem_iff.mp member)
  have sortedProduct : sorted.prod = value := permutation.prod_eq.trans productEq
  have sortedSize : effectiveSize sorted = size := by
    simpa only [effectiveSize, permutation.length_eq, permutation.prod_eq, permutation.sum_eq] using sizeEq
  have nonempty : sorted ≠ [] := by
    intro empty
    simp [empty, effectiveSize] at sortedSize
    omega
  have result := search_complete maxSize limit 2 [] sorted (by simp [Admissible]) sortedValid
    (List.pairwise_mergeSort' (· ≤ ·) factors) sortedValid (by omega) nonempty
    (by simpa [sortedProduct] using valueBound) (by simpa [sortedSize] using sizeBound)
  simpa [sortedSize, sortedProduct] using result

theorem root_iff (maxSize limit size value : Nat) (nontrivial : 2 ≤ size) :
    (size, value) ∈ search maxSize limit 2 1 0 0 ↔
      size ≤ maxSize ∧ value ≤ limit ∧ Represents size value := by
  constructor
  · exact search_sound maxSize limit 2 [] (by simp [Admissible]) (by omega) size value
  · rintro ⟨sizeBound, valueBound, represented⟩
    exact root_complete maxSize limit size value sizeBound nontrivial valueBound represented

end ProjectEulerSolutions.ProductSumNumbers
