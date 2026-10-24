import ProjectEulerSolutions.Lib.DigitMask
import Mathlib.Data.List.Nodup
import lang.py.builtin.type.int

/-! Finite subsets enumerated by the Python solver's integer bit masks. -/
namespace ProjectEulerSolutions.BitSubsets

set_option Elab.async false

def positions (bits width : Nat) : List Nat :=
  (List.range width).filter bits.testBit

theorem mem_positions (bits width index : Nat) :
    index ∈ positions bits width ↔ index < width ∧ bits.testBit index = true := by
  simp [positions]

theorem positions_nodup (bits width : Nat) : (positions bits width).Nodup :=
  List.nodup_range.filter _

/-- Every subset of the available positions has a mask inside the source loop's bound. -/
theorem mask_bound (indices : List Nat) (width : Nat) (bounded : ∀ index ∈ indices, index < width) :
    DigitMask.mask indices < 2 ^ width := by
  apply Nat.lt_pow_two_of_testBit
  intro index outside
  rw [DigitMask.testBit_mask]
  simp only [decide_eq_false_iff_not]
  intro member
  have := bounded index member
  omega

theorem positions_mask_perm (indices : List Nat) (width : Nat) (distinct : indices.Nodup)
    (bounded : ∀ index ∈ indices, index < width) :
    (positions (DigitMask.mask indices) width).Perm indices := by
  apply (List.perm_ext_iff_of_nodup (positions_nodup _ _) distinct).mpr
  intro index
  simp only [mem_positions, DigitMask.testBit_mask, decide_eq_true_eq]
  exact ⟨And.right, fun member => ⟨bounded index member, member⟩⟩

/-- Reconstructing an in-range mask from its selected positions loses no bits. -/
theorem mask_positions (bits width : Nat) (bounded : bits < 2 ^ width) :
    DigitMask.mask (positions bits width) = bits := by
  apply Nat.eq_of_testBit_eq
  intro index
  rw [DigitMask.testBit_mask]
  by_cases inside : index < width
  · simp only [mem_positions, inside, true_and, Bool.decide_eq_true]
  · have zero : bits.testBit index = false := Nat.testBit_lt_two_pow
      (bounded.trans_le (Nat.pow_le_pow_right (by decide) (by omega)))
    simp [mem_positions, inside, zero]

theorem positions_nonempty (bits width : Nat) (bounded : bits < 2 ^ width) :
    positions bits width ≠ [] ↔ bits ≠ 0 := by
  constructor
  · intro nonempty zero
    simp [zero, positions] at nonempty
  · intro nonzero empty
    have equal := mask_positions bits width bounded
    simp only [empty, DigitMask.mask] at equal
    exact nonzero equal.symm

private theorem positions_perm (bits left right : Nat) (leftBound : bits < 2 ^ left) (rightBound : bits < 2 ^ right) :
    (positions bits left).Perm (positions bits right) := by
  apply (List.perm_ext_iff_of_nodup (positions_nodup _ _) (positions_nodup _ _)).mpr
  intro index
  simp only [mem_positions]
  have below (width : Nat) (bound : bits < 2 ^ width) (set : bits.testBit index = true) : index < width := by
    by_contra outside
    have zero : bits.testBit index = false := Nat.testBit_lt_two_pow
      (bound.trans_le (Nat.pow_le_pow_right (by decide) (by omega)))
    simp only [set, Bool.true_eq_false] at zero
  exact ⟨fun selected => ⟨below right rightBound selected.2, selected.2⟩,
    fun selected => ⟨below left leftBound selected.2, selected.2⟩⟩

/-- The actual builtin `bit_count()` is the number of selected positions, for every mask width. -/
theorem bitCount_positions (bits width : Nat) (bounded : bits < 2 ^ width) :
    FormalSW.Py.Builtin.Int.bitCount bits = (positions bits width).length := by
  rw [FormalSW.Py.Builtin.Int.bitCount_eq_countP, List.countP_eq_length_filter]
  simp only [Int.natAbs_natCast]
  apply (positions_perm bits (FormalSW.Py.Builtin.Int.bitLength bits) width ?_ bounded).length_eq
  by_cases zero : bits = 0
  · simp [zero, FormalSW.Py.Builtin.Int.bitLength]
  · simpa [FormalSW.Py.Builtin.Int.bitLength, zero] using ((Nat.log2_eq_iff zero).mp rfl).2

/-- In particular, a mask constructed from a genuine subset has the subset's cardinality. -/
theorem bitCount_mask (indices : List Nat) (width : Nat) (distinct : indices.Nodup)
    (bounded : ∀ index ∈ indices, index < width) :
    FormalSW.Py.Builtin.Int.bitCount (DigitMask.mask indices) = indices.length := by
  rw [bitCount_positions _ width (mask_bound indices width bounded)]
  exact (positions_mask_perm indices width distinct bounded).length_eq

/-- Read the items at the set mask positions, in their original order. -/
def select (items : List Nat) (bits : Nat) : List Nat :=
  (positions bits items.length).map (fun index => items[index]!)

theorem select_subset (items : List Nat) (bits : Nat) : select items bits ⊆ items := by
  intro value member
  obtain ⟨index, selected, rfl⟩ := List.mem_map.mp member
  have inside := (mem_positions bits items.length index).mp selected |>.1
  rw [List.getElem!_of_getElem? (List.getElem?_eq_getElem inside)]
  exact List.getElem_mem inside

theorem select_nodup (items : List Nat) (distinct : items.Nodup) (bits : Nat) :
    (select items bits).Nodup := by
  apply List.Nodup.map_on ?_ (positions_nodup _ _)
  intro left leftMember right rightMember equal
  exact (List.getElem!_inj (mem_positions _ _ _ |>.mp leftMember).1
    (mem_positions _ _ _ |>.mp rightMember).1 distinct).mp equal

theorem select_length (items : List Nat) (bits : Nat) (bounded : bits < 2 ^ items.length) :
    (select items bits).length = FormalSW.Py.Builtin.Int.bitCount bits := by
  simpa only [select, List.length_map] using (bitCount_positions bits items.length bounded).symm

theorem select_nonempty (items : List Nat) (bits : Nat) (bounded : bits < 2 ^ items.length) :
    select items bits ≠ [] ↔ bits ≠ 0 := by
  simpa only [select, ne_eq, List.map_eq_nil_iff] using positions_nonempty bits items.length bounded

/-- Every genuine subset of a digit-position group is reached by the solver's mask enumeration. -/
theorem select_complete (items chosen : List Nat) (distinct : chosen.Nodup) (included : chosen ⊆ items) :
    ∃ bits < 2 ^ items.length, (select items bits).Perm chosen ∧
      FormalSW.Py.Builtin.Int.bitCount bits = chosen.length := by
  let slots := chosen.map items.idxOf
  have bounded : ∀ index ∈ slots, index < items.length := by
    intro index member
    obtain ⟨value, present, rfl⟩ := List.mem_map.mp member
    exact List.idxOf_lt_length_iff.mpr (included present)
  have unique : slots.Nodup := by
    apply List.Nodup.map_on ?_ distinct
    intro left leftMember right _ equal
    exact (List.idxOf_inj (included leftMember)).mp equal
  let bits := DigitMask.mask slots
  refine ⟨bits, mask_bound slots items.length bounded, ?_, ?_⟩
  · have mapped := (positions_mask_perm slots items.length unique bounded).map (fun index => items[index]!)
    have recovered : slots.map (fun index => items[index]!) = chosen := by
      simp only [slots, List.map_map]
      calc
        _ = chosen.map id := by
          apply List.map_congr_left
          intro value member
          exact List.getElem!_of_getElem? (List.getElem?_idxOf (included member))
        _ = chosen := List.map_id chosen
    simpa only [select, recovered] using mapped
  · simpa only [slots, List.length_map] using bitCount_mask slots items.length unique bounded

end ProjectEulerSolutions.BitSubsets
