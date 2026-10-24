import ProjectEulerSolutions.Lib.PandigitalDigits
import ProjectEulerStatements.Lib.ConcatenatedProductStrings

/-! Complete concatenated-product searches for a variable target digit count. -/
namespace ProjectEulerSolutions.PandigitalConcat
open ProjectEulerStatements.P38 ProjectEulerStatements.ConcatenatedProducts
set_option Elab.async false

/-- Two products must fit in the target width. -/
def seedLimit (width : Nat) : Nat := 10 ^ (width / 2)

def accepted (width n k : Nat) : Bool :=
  decide (1 < k) && PandigitalDigits.check width (digitsBE (concatProduct n k))

theorem accepted_spec (width n k : Nat) :
    accepted width n k = true ↔ 1 < k ∧
      (digitsBE (concatProduct n k)).Perm (List.range' 1 width) := by
  simp only [accepted, Bool.and_eq_true, decide_eq_true_eq, PandigitalDigits.check_iff_perm]

theorem accepted_length (width n k : Nat) (yes : accepted width n k = true) :
    numDigits (concatProduct n k) = width := by
  have digits := ((accepted_spec width n k).mp yes).2.length_eq
  simpa only [digitsBE, List.length_reverse, List.length_range', numDigits] using digits

theorem length_lower (n k : Nat) :
    k * numDigits n ≤ numDigits (concatProduct n k) := by
  induction k with
  | zero => simp [concatProduct_zero, numDigits]
  | succ k ih =>
      have product : n ≤ n * (k + 1) := by nlinarith
      have digits := Nat.le_length_digits_le 10 n (n * (k + 1)) product
      rw [concatProduct_length_succ]
      change (k + 1) * numDigits n ≤ numDigits (concatProduct n k) + (Nat.digits 10 (n * (k + 1))).length
      dsimp [numDigits] at ih ⊢
      nlinarith

/-- Every successful seed lies below the input-derived endpoint, regardless of its number of products. -/
theorem seed_bounded (width n k : Nat) (yes : accepted width n k = true) :
    n < seedLimit width := by
  have count := ((accepted_spec width n k).mp yes).1
  have lower := length_lower n k
  rw [accepted_length width n k yes] at lower
  have doubled : numDigits n * 2 ≤ width := by nlinarith
  have digits : numDigits n ≤ width / 2 := (Nat.le_div_iff_mul_le (by decide)).mpr doubled
  exact (Nat.digits_length_le_iff (by decide : 1 < 10) n).mp digits

theorem count_bounded (width n k : Nat) (positive : 0 < n) (yes : accepted width n k = true) :
    k ≤ width := by
  have lower := concatProduct_length_lower n k positive
  rw [accepted_length width n k yes] at lower
  exact lower

private theorem width_reached (width n : Nat) (positive : 0 < n) :
    ∃ k, width ≤ numDigits (concatProduct n k) := ⟨width, concatProduct_length_lower n width positive⟩

/-- The first product count at which the concatenation reaches the requested width. -/
noncomputable def stop (width n : Nat) : Nat :=
  if positive : 0 < n then Nat.find (width_reached width n positive) else 0

theorem stop_spec (width n : Nat) (positive : 0 < n) :
    width ≤ numDigits (concatProduct n (stop width n)) := by
  simpa only [stop, dif_pos positive] using Nat.find_spec (width_reached width n positive)

theorem before_stop (width n k : Nat) (positive : 0 < n) (before : k < stop width n) :
    numDigits (concatProduct n k) < width := by
  simp only [stop, dif_pos positive] at before
  have missing := Nat.find_min (width_reached width n positive) before
  omega

theorem stop_bounds (width n : Nat) (nonempty : 0 < width) (positive : 0 < n) :
    1 ≤ stop width n ∧ stop width n ≤ width := by
  have reached := stop_spec width n positive
  have lower : 0 < stop width n := by
    by_contra absent
    have zero : stop width n = 0 := by omega
    simp [zero, concatProduct_zero, numDigits] at reached
    omega
  have upper : stop width n ≤ width := by
    by_contra late
    have before := before_stop width n width positive (by omega)
    have length := concatProduct_length_lower n width positive
    omega
  omega

theorem exact_width_unique (width n k : Nat) (positive : 0 < n)
    (same : numDigits (concatProduct n k) = width) : k = stop width n := by
  have reached := stop_spec width n positive
  rcases lt_trichotomy k (stop width n) with before | equal | after
  · have missing := before_stop width n k positive before
    omega
  · exact equal
  · have increasing := concatProduct_length_strict n positive after
    dsimp only at increasing
    omega

noncomputable def seedScore (width n : Nat) : Nat :=
  if accepted width n (stop width n) then concatProduct n (stop width n) else 0

def seedValues (width n : Nat) : List Nat :=
  (List.range' 2 (width - 1)).filterMap fun k =>
    if accepted width n k then some (concatProduct n k) else none

private theorem select_unique (items : List Nat) (unique : items.Nodup) (target value : Nat) :
    (items.filterMap fun i => if i = target then some value else none) =
      if target ∈ items then [value] else [] := by
  induction items with
  | nil => simp
  | cons head tail ih =>
      obtain ⟨absent, tailUnique⟩ := List.nodup_cons.mp unique
      by_cases same : head = target
      · subst head
        simp [absent, ih tailUnique]
      · simp [same, Ne.symm same, ih tailUnique]

theorem seedValues_at_stop (width n : Nat) (nonempty : 0 < width) (positive : 0 < n) :
    seedValues width n = if accepted width n (stop width n) then [concatProduct n (stop width n)] else [] := by
  have bounds := stop_bounds width n nonempty positive
  by_cases yes : accepted width n (stop width n) = true
  · have replace (k : Nat) :
        (if accepted width n k then some (concatProduct n k) else none) =
          if k = stop width n then some (concatProduct n (stop width n)) else none := by
      by_cases same : k = stop width n
      · simp [same, yes]
      · have absent : accepted width n k = false := by
          apply Bool.eq_false_iff.mpr
          intro hit
          exact same (exact_width_unique width n k positive (accepted_length width n k hit))
        simp [same, absent]
    simp only [seedValues, replace, if_pos yes]
    rw [select_unique _ (List.nodup_range')]
    have two := ((accepted_spec width n (stop width n)).mp yes).1
    have member : stop width n ∈ List.range' 2 (width - 1) :=
      List.mem_range'.mpr ⟨stop width n - 2, by omega, by omega⟩
    simp only [if_pos member]
  · have absent (k : Nat) : accepted width n k = false := by
      apply Bool.eq_false_iff.mpr
      intro hit
      have same := exact_width_unique width n k positive (accepted_length width n k hit)
      exact yes (same ▸ hit)
    simp [seedValues, absent]

noncomputable def scan (width first count best : Nat) : Nat :=
  (List.range' first count).foldl (fun best n => max best (seedScore width n)) best

def candidates (width : Nat) : List Nat :=
  (List.range' 1 (seedLimit width - 1)).flatMap (seedValues width)

def reference (width : Nat) : Nat := (candidates width).foldl Nat.max 0

/-- Decimal numbers cannot contain the required digit ten when the target is wider than nine. -/
theorem accepted_large (width n k : Nat) (large : 9 < width) : accepted width n k = false := by
  apply Bool.eq_false_iff.mpr
  intro yes
  have permutation := ((accepted_spec width n k).mp yes).2
  have target : 10 ∈ List.range' 1 width := List.mem_range'.mpr ⟨9, by omega, by omega⟩
  have member := permutation.mem_iff.mpr target
  have decimal := Nat.digits_lt_base (by decide : 1 < 10) (List.mem_reverse.mp member)
  omega

theorem reference_large (width : Nat) (large : 9 < width) : reference width = 0 := by
  have empty (n : Nat) : seedValues width n = [] := by
    simp [seedValues, accepted_large width n _ large]
  have noCandidates : candidates width = [] := List.flatMap_eq_nil_iff.mpr (fun n _ => empty n)
  simp [reference, noCandidates]

theorem reference_small (width : Nat) (small : width < 2) : reference width = 0 := by
  have quotient : width / 2 = 0 := by omega
  simp [reference, candidates, seedLimit, quotient]

theorem seed_fold (width n best : Nat) (nonempty : 0 < width) (positive : 0 < n) :
    (seedValues width n).foldl Nat.max best = max best (seedScore width n) := by
  rw [seedValues_at_stop width n nonempty positive]
  cases yes : accepted width n (stop width n) <;> simp [yes, seedScore]

theorem candidates_fold (width : Nat) (nonempty : 0 < width) (items : List Nat) (best : Nat)
    (positive : ∀ n ∈ items, 0 < n) :
    (items.flatMap (seedValues width)).foldl Nat.max best =
      items.foldl (fun best n => max best (seedScore width n)) best := by
  induction items generalizing best with
  | nil => rfl
  | cons head tail ih =>
      rw [List.flatMap_cons, List.foldl_append, seed_fold width head best nonempty (positive head (by simp))]
      exact ih _ (fun n member => positive n (by simp [member]))

theorem reference_scan (width : Nat) (nonempty : 0 < width) :
    reference width = scan width 1 (seedLimit width - 1) 0 := by
  apply candidates_fold width nonempty
  intro n member
  obtain ⟨offset, _, same⟩ := List.mem_range'.mp member
  omega

theorem scan_zero (width first best : Nat) : scan width first 0 best = best := rfl

theorem scan_succ (width first count best : Nat) :
    scan width first (count + 1) best = scan width (first + 1) count (max best (seedScore width first)) := by
  simp only [scan, List.range'_succ, List.foldl_cons]

/-- A solution is any positive seed with at least two products whose digits are a permutation of the target. -/
def Solution (width value : Nat) : Prop :=
  ∃ n k, 0 < n ∧ accepted width n k = true ∧ concatProduct n k = value

/-- The finite candidate list contains every solution, without an external seed or product-count cutoff. -/
theorem mem_candidates (width value : Nat) (nonempty : 0 < width) :
    value ∈ candidates width ↔ Solution width value := by
  simp only [candidates, List.mem_flatMap, seedValues, List.mem_filterMap]
  constructor
  · rintro ⟨n, seed, k, count, selected⟩
    have positive : 0 < n := by
      obtain ⟨offset, _, same⟩ := List.mem_range'.mp seed
      omega
    split at selected
    next accepted => exact ⟨n, k, positive, accepted, Option.some.inj selected⟩
    next => cases selected
  · rintro ⟨n, k, positive, yes, same⟩
    have seed := seed_bounded width n k yes
    have count := count_bounded width n k positive yes
    have two := ((accepted_spec width n k).mp yes).1
    refine ⟨n, List.mem_range'.mpr ⟨n - 1, by omega, by omega⟩,
      k, List.mem_range'.mpr ⟨k - 2, by omega, by omega⟩, ?_⟩
    simp only [yes, if_true, same]

end ProjectEulerSolutions.PandigitalConcat
