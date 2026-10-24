import ProjectEulerSolutions.Lib.PrimeReplacementPruning
import ProjectEulerSolutions.Lib.BitSubsets
import Mathlib.Data.List.Sublists

/-! Completeness of shared-digit grouping and the large-family search restrictions. -/
namespace ProjectEulerSolutions.PrimeReplacementSearch
open ProjectEulerStatements.P51 PrimeReplacement

set_option Elab.async false

/-- Any nonempty selection of equal digits with enough prime replacements is a reference witness. -/
theorem family_of_selection (target n original : Nat) (indices : List Nat) (prime : Nat.Prime n)
    (nonempty : indices ≠ []) (bounded : ∀ index ∈ indices, index < (Nat.digits 10 n).length)
    (same : SameDigit (Nat.digits 10 n).reverse indices original)
    (large : target ≤ primeFamilySize n (reflected (Nat.digits 10 n).length indices)) :
    hasPrimeFamily target n = true := by
  let raw := reflected (Nat.digits 10 n).length indices
  let canonical := (List.range (Nat.digits 10 n).length).filter (fun index => decide (index ∈ raw))
  have rawBounded := reflected_bounded (Nat.digits 10 n).length indices bounded
  have equivalent (index : Nat) : index ∈ canonical ↔ index ∈ raw := by
    simp only [canonical, List.mem_filter, List.mem_range, decide_eq_true_eq]
    exact ⟨And.right, fun member => ⟨rawBounded index member, member⟩⟩
  have rawNonempty : raw ≠ [] := by
    simpa only [raw, reflected, ne_eq, List.map_eq_nil_iff] using nonempty
  have canonicalNonempty : canonical ≠ [] := by
    intro empty
    obtain ⟨index, member⟩ := List.exists_mem_of_ne_nil raw rawNonempty
    have := (equivalent index).mpr member
    simp only [empty, List.not_mem_nil] at this
  have families := family_congr_indices n canonical raw equivalent
  have scores := familySize_congr_indices n canonical raw equivalent
  have originalMember : n ∈ familyNumbers n canonical := by
    rw [families]
    exact family_member_of_sameDigit n raw original ((sameDigit_reflected _ _ _ bounded).mp same)
  rw [hasPrimeFamily, Bool.and_eq_true]
  refine ⟨List.any_eq_true.mpr ⟨canonical, ?_, ?_⟩, by simpa only [isPrime, decide_eq_true_eq] using prime⟩
  · exact List.mem_filter.mpr ⟨List.mem_sublists.mpr List.filter_sublist, by simpa using canonicalNonempty⟩
  · simp only [Bool.and_eq_true, decide_eq_true_eq]
    exact ⟨originalMember, by simpa only [scores] using large⟩

/-- Every reference witness has an equal-digit selection that survives the solver's pruning. -/
theorem selection_of_family (target n : Nat) (family : hasPrimeFamily target n = true) :
    ∃ original < 10, ∃ indices : List Nat,
      indices.Nodup ∧ indices ≠ [] ∧
      (∀ index ∈ indices, index < (Nat.digits 10 n).length) ∧
      SameDigit (Nat.digits 10 n).reverse indices original ∧
      target ≤ primeFamilySize n (reflected (Nat.digits 10 n).length indices) ∧
      (8 ≤ target → (Nat.digits 10 n).length - 1 ∉ indices ∧
        3 ≤ indices.length ∧ indices.length % 3 = 0) := by
  rw [hasPrimeFamily, Bool.and_eq_true] at family
  obtain ⟨raw, member, accepted⟩ := List.any_eq_true.mp family.1
  obtain ⟨subsets, nonempty⟩ := List.mem_filter.mp member
  have sublist := List.mem_sublists.mp subsets
  have distinct : raw.Nodup := List.nodup_range.sublist sublist
  have rawNonempty : raw ≠ [] := by simpa only [decide_eq_true_eq] using nonempty
  have rawBounded : ∀ index ∈ raw, index < (Nat.digits 10 n).length :=
    fun index present => List.mem_range.mp (sublist.subset present)
  simp only [Bool.and_eq_true, decide_eq_true_eq] at accepted
  have self : n ∈ familyNumbers n raw := accepted.1
  have score : target ≤ primeFamilySize n raw := accepted.2
  obtain ⟨original, digitBound, same⟩ := (family_member_iff n raw).mp self
  let indices := reflected (Nat.digits 10 n).length raw
  have bounded := reflected_bounded (Nat.digits 10 n).length raw rawBounded
  have twice := reflected_twice (Nat.digits 10 n).length raw rawBounded
  refine ⟨original, digitBound, indices, reflected_nodup _ raw distinct rawBounded, ?_, bounded, ?_, ?_, ?_⟩
  · simpa only [indices, reflected, ne_eq, List.map_eq_nil_iff] using rawNonempty
  · apply (sameDigit_reflected (Nat.digits 10 n) indices original bounded).mpr
    simpa only [indices, twice] using same
  · simpa only [indices, twice] using score
  · intro threshold
    have large : 7 < primeFamilySize n raw := by omega
    have units := PrimeReplacementPruning.large_family_excludes_units n raw (by omega)
    have multiple := PrimeReplacementPruning.large_family_count_multiple_three n raw distinct rawNonempty rawBounded self large
    have minimum := PrimeReplacementPruning.large_family_min_positions n raw distinct rawNonempty rawBounded self large
    have positive : 0 < (Nat.digits 10 n).length := by
      obtain ⟨index, present⟩ := List.exists_mem_of_ne_nil raw rawNonempty
      have := rawBounded index present
      omega
    refine ⟨?_, by simpa only [indices, reflected, List.length_map] using minimum,
      by simpa only [indices, reflected, List.length_map] using multiple⟩
    simpa only [indices, leading_reflected _ positive raw rawBounded] using units

/-- The source groups matching decimal positions, omitting units when searching for eight primes. -/
def group (target n original : Nat) : List Nat :=
  (List.range (if 8 ≤ target then (Nat.digits 10 n).length - 1 else (Nat.digits 10 n).length)).filter
    (fun index => decide ((Nat.digits 10 n).reverse[index]? = some original))

private theorem group_bounded (target n original : Nat) :
    ∀ index ∈ group target n original, index < (Nat.digits 10 n).length := by
  intro index member
  have inside := List.mem_range.mp (List.mem_filter.mp member).1
  split_ifs at inside <;> omega

private theorem group_same (target n original : Nat) :
    SameDigit (Nat.digits 10 n).reverse (group target n original) original := by
  intro index value found member
  have equal := of_decide_eq_true (List.mem_filter.mp member).2
  simp only [Nat.zero_add] at equal
  rw [found] at equal
  exact Option.some.inj equal

/-- Every qualifying reference family is reached by a surviving digit-group mask. -/
theorem mask_of_family (target n : Nat) (family : hasPrimeFamily target n = true) :
    ∃ original < 10, ∃ bits : Nat,
      1 ≤ bits ∧ bits < 2 ^ (group target n original).length ∧
      (8 ≤ target → 3 ≤ (group target n original).length ∧ FormalSW.Py.Builtin.Int.bitCount bits % 3 = 0) ∧
      target ≤ primeFamilySize n (reflected (Nat.digits 10 n).length (BitSubsets.select (group target n original) bits)) := by
  obtain ⟨original, digitBound, indices, distinct, nonempty, bounded, same, score, pruning⟩ :=
    selection_of_family target n family
  let items := group target n original
  have included : indices ⊆ items := by
    intro index member
    have inside := bounded index member
    have reverseInside : index < (Nat.digits 10 n).reverse.length := by simpa only [List.length_reverse] using inside
    have read := List.getElem?_eq_getElem reverseInside
    have equal := same index (Nat.digits 10 n).reverse[index] read (by simpa only [Nat.zero_add] using member)
    refine List.mem_filter.mpr ⟨List.mem_range.mpr ?_, ?_⟩
    · split_ifs with threshold
      · have omitted := (pruning threshold).1
        have different : index ≠ (Nat.digits 10 n).length - 1 := fun equal => omitted (equal ▸ member)
        omega
      · exact inside
    · simp only [read, equal, decide_true]
  obtain ⟨bits, maskBound, selected, cardinality⟩ := BitSubsets.select_complete items indices distinct included
  have notEmpty : BitSubsets.select items bits ≠ [] := by
    intro empty
    have zero : indices.length = 0 := by rw [← selected.length_eq, empty]; rfl
    exact nonempty (List.length_eq_zero_iff.mp zero)
  have nonzero := (BitSubsets.select_nonempty items bits maskBound).mp notEmpty
  refine ⟨original, digitBound, bits, by omega, maskBound, ?_, ?_⟩
  · intro threshold
    change 3 ≤ items.length ∧ _
    have conditions := pruning threshold
    have sizeBound : FormalSW.Py.Builtin.Int.bitCount bits ≤ items.length := by
      rw [BitSubsets.bitCount_positions bits items.length maskBound]
      exact (List.length_filter_le _ _).trans_eq List.length_range
    exact ⟨by omega, by simpa only [cardinality] using conditions.2.2⟩
  · have reflectedSelection := selected.map (fun index => (Nat.digits 10 n).length - 1 - index)
    have equal := familySize_congr_indices n
      (reflected (Nat.digits 10 n).length (BitSubsets.select items bits))
      (reflected (Nat.digits 10 n).length indices) (fun index => reflectedSelection.mem_iff)
    change target ≤ primeFamilySize n (reflected (Nat.digits 10 n).length (BitSubsets.select items bits))
    rw [equal]
    exact score

/-- A family found by the digit-group enumeration is a reference witness. -/
theorem family_of_mask (target n original bits : Nat) (prime : Nat.Prime n)
    (nonzero : bits ≠ 0) (bounded : bits < 2 ^ (group target n original).length)
    (large : target ≤ primeFamilySize n (reflected (Nat.digits 10 n).length
      (BitSubsets.select (group target n original) bits))) : hasPrimeFamily target n = true := by
  let indices := BitSubsets.select (group target n original) bits
  have included := BitSubsets.select_subset (group target n original) bits
  apply family_of_selection target n original indices prime
    ((BitSubsets.select_nonempty _ _ bounded).mpr nonzero)
    (fun index member => group_bounded target n original index (included member)) ?_ large
  intro index value found member
  exact group_same target n original index value found (included member)

/-- The grouped-mask search with both pruning rules is equivalent to the reference family predicate. -/
theorem hasPrimeFamily_iff_mask (target n : Nat) : hasPrimeFamily target n = true ↔
    Nat.Prime n ∧ ∃ original < 10, ∃ bits : Nat,
      1 ≤ bits ∧ bits < 2 ^ (group target n original).length ∧
      (8 ≤ target → 3 ≤ (group target n original).length ∧ FormalSW.Py.Builtin.Int.bitCount bits % 3 = 0) ∧
      target ≤ primeFamilySize n (reflected (Nat.digits 10 n).length (BitSubsets.select (group target n original) bits)) := by
  constructor
  · intro family
    have accepted := family
    simp only [hasPrimeFamily, Bool.and_eq_true, isPrime, decide_eq_true_eq] at accepted
    exact ⟨accepted.2, mask_of_family target n family⟩
  · rintro ⟨prime, original, _, bits, positive, bounded, _, large⟩
    exact family_of_mask target n original bits prime (by omega) bounded large

end ProjectEulerSolutions.PrimeReplacementSearch
