import ProjectEulerStatements.Lib.PythagoreanPerimeterBounds

/-! A perimeter table allocated from the requested bound, with proved in-range updates. -/
namespace ProjectEulerSolutions.PythagoreanArraySieve
open ProjectEulerStatements.PythagoreanPerimeterCounting
set_option Elab.async false

def encode (size : Nat) (counts : Table) : Array Nat :=
  Array.ofFn (fun index : Fin size => counts index)

@[simp] theorem encode_size (size : Nat) (counts : Table) : (encode size counts).size = size := by
  simp [encode]

theorem encode_toList (size : Nat) (counts : Table) :
    (encode size counts).toList = (List.range size).map counts := by
  apply List.ext_getElem
  · simp
  · intro index leftBound rightBound
    simp [encode]

def bump (cells : Array Nat) (index : Nat) : Array Nat :=
  if inside : index < cells.size then cells.set index (cells[index] + 1) else cells

theorem bump_encode (size index : Nat) (counts : Table) (inside : index < size) :
    bump (encode size counts) index = encode size (increment counts index) := by
  apply Array.ext
  · simp [bump, inside]
  · intro query leftBound rightBound
    have bounded : query < size := by simpa only [encode_size] using rightBound
    simp only [bump, encode_size, dif_pos inside, Array.getElem_set]
    by_cases same : index = query
    · subst query
      simp [encode, increment]
    · simp [encode, increment, same, Ne.symm same]

def multiples (limit step p : Nat) (positive : 0 < step) (cells : Array Nat) : Array Nat :=
  if _inside : p ≤ limit then multiples limit step (p + step) positive (bump cells p)
  else cells
termination_by limit + 1 - p
decreasing_by omega

def rowArray (limit m n : Nat) (positive : 0 < m) (cells : Array Nat) : Array Nat :=
  if ordered : n < m then
    if ((m ^^^ n) &&& 1) = 0 then rowArray limit m (n + 1) positive cells
    else if Nat.gcd m n ≠ 1 then rowArray limit m (n + 1) positive cells
    else if limit < perimeter m n then cells
    else rowArray limit m (n + 1) positive
      (multiples limit (perimeter m n) (perimeter m n)
        (by unfold perimeter; positivity) cells)
  else cells
termination_by m - n
decreasing_by all_goals omega

def outerArray (limit m : Nat) (positive : 2 ≤ m) (cells : Array Nat) : Array Nat :=
  if inside : m ≤ maxParameter limit then
    outerArray limit (m + 1) (by omega) (rowArray limit m 1 (by omega) cells)
  else cells
termination_by maxParameter limit + 1 - m
decreasing_by omega

def compute (limit : Nat) : Array Nat :=
  outerArray limit 2 (by omega) (Array.replicate (limit + 1) 0)

theorem multiples_encode (limit m n first count : Nat) (counts : Table)
    (positive : 0 < perimeter m n) (endpoint : first + count = limit / perimeter m n + 1) :
    multiples limit (perimeter m n) (first * perimeter m n) positive (encode (limit + 1) counts) =
      encode (limit + 1) (addMultiples limit m n first count counts) := by
  induction count generalizing first counts with
  | zero =>
      have outside : limit < first * perimeter m n := by
        have less := Nat.lt_mul_div_succ limit positive
        rw [Nat.mul_comm] at less
        have equal : first = limit / perimeter m n + 1 := by omega
        simpa only [equal] using less
      rw [multiples, dif_neg (not_le.mpr outside), addMultiples_zero]
  | succ count ih =>
      have inside : first * perimeter m n ≤ limit :=
        (Nat.le_div_iff_mul_le positive).mp (by omega)
      rw [multiples, dif_pos inside, bump_encode _ _ _ (by omega), addMultiples_step]
      simpa only [Nat.add_mul, Nat.one_mul] using ih (first + 1)
        (increment counts (first * perimeter m n)) (by omega)

theorem rowArray_encode (limit m first count : Nat) (counts : Table)
    (positive : 0 < m) (endpoint : first + count = m) :
    rowArray limit m first positive (encode (limit + 1) counts) =
      encode (limit + 1) (row limit m first count counts) := by
  induction count generalizing first counts with
  | zero =>
      have stopped : ¬ first < m := by omega
      rw [rowArray, dif_neg stopped, row_zero]
  | succ count ih =>
      have ordered : first < m := by omega
      rw [rowArray, dif_pos ordered]
      by_cases parity : ((m ^^^ first) &&& 1) = 0
      · rw [if_pos parity, row_step]
        have even : (m - first) % 2 = 0 := by
          simpa only [xor_parity m first ordered.le] using parity
        have rejected : permitted limit m first = false := by simp [permitted, even]
        simp only [rejected, Bool.false_eq_true, if_false]
        exact ih (first + 1) counts (by omega)
      · rw [if_neg parity]
        have odd : (m - first) % 2 = 1 := by
          rw [xor_parity m first ordered.le] at parity
          omega
        by_cases divisor : Nat.gcd m first ≠ 1
        · rw [if_pos divisor, row_step]
          have rejected : permitted limit m first = false := by
            simp [permitted, divisor]
          simp only [rejected, Bool.false_eq_true, if_false]
          exact ih (first + 1) counts (by omega)
        · rw [if_neg divisor]
          by_cases tooLarge : limit < perimeter m first
          · rw [if_pos tooLarge, row_inactive _ _ _ _ _ tooLarge]
          · rw [if_neg tooLarge, row_step]
            have accepted : permitted limit m first = true := by
              simp [permitted, odd, Nat.coprime_iff_gcd_eq_one,
                not_ne_iff.mp divisor, not_lt.mp tooLarge]
            simp only [accepted, if_true]
            have perimeterPositive : 0 < perimeter m first := by unfold perimeter; positivity
            have updated := multiples_encode limit m first 1 (limit / perimeter m first)
              counts perimeterPositive (by omega)
            simp only [Nat.one_mul] at updated
            rw [updated]
            exact ih (first + 1) _ (by omega)

theorem outerArray_encode (limit first count : Nat) (counts : Table)
    (positive : 2 ≤ first) (endpoint : first + count = maxParameter limit + 1) :
    outerArray limit first positive (encode (limit + 1) counts) =
      encode (limit + 1) (outer limit first count counts) := by
  induction count generalizing first counts with
  | zero =>
      have stopped : ¬ first ≤ maxParameter limit := by omega
      rw [outerArray, dif_neg stopped, outer_zero]
  | succ count ih =>
      have inside : first ≤ maxParameter limit := by omega
      rw [outerArray, dif_pos inside, rowArray_encode _ _ _ (first - 1) _ (by omega) (by omega),
        outer_step]
      by_cases active : 2 * first * (first + 1) ≤ limit
      · rw [if_pos active]
        exact ih (first + 1) _ (by omega) (by omega)
      · have inactive : row limit first 1 (first - 1) counts = counts := by
          apply row_inactive
          simpa only [perimeter] using not_le.mp active
        rw [if_neg active, inactive]
        exact ih (first + 1) counts (by omega) (by omega)

/-- The executable array contains the complete triangle multiplicity at every in-range perimeter. -/
theorem compute_encode (limit : Nat) : compute limit = encode (limit + 1) (table limit) := by
  have initialized : Array.replicate (limit + 1) 0 = encode (limit + 1) (fun _ => 0) := by
    apply Array.ext
    · simp
    · intro index leftBound rightBound
      simp [encode]
  rw [compute, initialized, outerArray_encode limit 2 (Nat.sqrt (limit / 2)) _ (by omega)
    (by dsimp only [maxParameter]; omega), table_at_sqrt]

end ProjectEulerSolutions.PythagoreanArraySieve
