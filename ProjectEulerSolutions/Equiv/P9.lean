import ProjectEulerSolutions.P9
import Mathlib.NumberTheory.PythagoreanTriples
import Mathlib.Tactic
namespace ProjectEulerSolutions.P9

lemma loopN_eq_fold (total m n best : Nat) :
    loopN total m n best =
      (List.range' n (m - n)).foldl (fun acc x => Nat.max acc (candidate total m x)) best := by
  induction n, best using loopN.induct total m with
  | case1 n best h =>
      rw [loopN.eq_def]
      simp [h]
  | case2 n best h ih =>
      rw [loopN.eq_def]
      simp [h]
      rw [ih]
      have hmn : m - n = (m - (n + 1)) + 1 := by omega
      rw [hmn, List.range'_succ]
      simp [List.foldl_cons]

lemma loopM_stop_mono {total m x : Nat} (hstop : 2 * m * (m + 1) > total) (hmx : m ≤ x) :
    2 * x * (x + 1) > total := by
  have hle1 : m + 1 ≤ x + 1 := by omega
  have hle : 2 * m * (m + 1) ≤ 2 * x * (x + 1) := by
    nlinarith [Nat.mul_le_mul hmx hle1]
  omega

lemma range'_ge_start {start len x : Nat} (hx : x ∈ List.range' start len) : start ≤ x := by
  rw [List.mem_range'] at hx
  obtain ⟨i, _hi, rfl⟩ := hx
  omega

lemma foldl_if_pos_eq_self (total : Nat) (l : List Nat) (best : Nat)
    (h : ∀ x ∈ l, 2 * x * (x + 1) > total) :
    l.foldl (fun acc x =>
      if 2 * x * (x + 1) > total then acc else Nat.max acc (loopN total x 1 0)) best = best := by
  induction l generalizing best with
  | nil => simp
  | cons x xs ih =>
      have hx : 2 * x * (x + 1) > total := h x (by simp)
      have hxs : ∀ y ∈ xs, 2 * y * (y + 1) > total := by
        intro y hy
        exact h y (by simp [hy])
      simp [hx, ih best hxs]

lemma loopM_eq_fold (total m best : Nat) :
    loopM total m best =
      (List.range' m (total + 1 - m)).foldl
        (fun acc x =>
          if 2 * x * (x + 1) > total then acc else Nat.max acc (loopN total x 1 0)) best := by
  induction m, best using loopM.induct total with
  | case1 m best hstop =>
      rw [loopM.eq_def]
      simp [hstop]
      exact (foldl_if_pos_eq_self total (List.range' m (total + 1 - m)) best (by
        intro x hx
        exact loopM_stop_mono hstop (range'_ge_start hx))).symm
  | case2 m best hstop ih =>
      rw [loopM.eq_def]
      simp [hstop]
      rw [ih]
      have hm_le : m ≤ total := by
        have hmul : m ≤ 2 * m * (m + 1) := by nlinarith
        exact le_trans hmul (le_of_not_gt hstop)
      have hlen : total + 1 - m = (total + 1 - (m + 1)) + 1 := by omega
      rw [hlen, List.range'_succ]
      simp [List.foldl_cons, hstop]

lemma candidate_mem_tripletProducts_of_ne_zero {total m n : Nat}
    (h : candidate total m n ≠ 0) :
    candidate total m n ∈ ProjectEulerStatements.P9.tripletProducts total := by
  unfold candidate at h ⊢
  by_cases hv : candidateValid total m n
  · simp [hv]
    unfold candidateValid at hv
    let a := euclidA total m n
    let b := euclidB total m n
    let c := euclidC total m n
    change a * b * c ∈ ProjectEulerStatements.P9.tripletProducts total
    have hprod : a * b * c ≠ 0 := by
      intro hz
      exact h (by simp [a, b, c, hz])
    have hb0' : 0 < b := by
      apply Nat.pos_of_ne_zero
      intro hb
      apply hprod
      simp [hb]
    rcases hv with ⟨_hmod, ha0, horder, hpyt, hsum⟩
    have ha0' : 0 < a := by simpa [a] using ha0
    have hpyt_mul : a * a + b * b = c * c := by simpa [a, b, c] using hpyt
    have hpyt_pow : a ^ 2 + b ^ 2 = c ^ 2 := by simpa [pow_two] using hpyt_mul
    have hpyt_swap_pow : b ^ 2 + a ^ 2 = c ^ 2 := by
      rw [Nat.add_comm]
      exact hpyt_pow
    have hsum' : a + b + c = total := by simpa [a, b, c] using hsum
    unfold ProjectEulerStatements.P9.tripletProducts
    rcases horder with ⟨hab, hbc⟩ | ⟨hba, hac⟩
    · have hab' : a < b := by simpa [a, b] using hab
      have hbc' : b < c := by simpa [b, c] using hbc
      apply Finset.mem_image.mpr
      refine ⟨(a, b), ?_, ?_⟩
      · rw [Finset.mem_filter]
        constructor
        · simp
          omega
        · change a < b ∧ b < total - a - b ∧ a ^ 2 + b ^ 2 = (total - a - b) ^ 2
          rw [show total - a - b = c by omega]
          exact ⟨hab', hbc', hpyt_pow⟩
      · change a * b * (total - a - b) = a * b * c
        rw [show total - a - b = c by omega]
    · have hba' : b < a := by simpa [a, b] using hba
      have hac' : a < c := by simpa [a, c] using hac
      apply Finset.mem_image.mpr
      refine ⟨(b, a), ?_, ?_⟩
      · rw [Finset.mem_filter]
        constructor
        · simp
          omega
        · change b < a ∧ a < total - b - a ∧ b ^ 2 + a ^ 2 = (total - b - a) ^ 2
          rw [show total - b - a = c by omega]
          exact ⟨hba', hac', hpyt_swap_pow⟩
      · change b * a * (total - b - a) = a * b * c
        rw [show total - b - a = c by omega]
        ring
  · simp [hv] at h

lemma mem_tripletProducts_le_naive {total x : Nat}
    (hx : x ∈ ProjectEulerStatements.P9.tripletProducts total) :
    x ≤ ProjectEulerStatements.P9.naive total := by
  unfold ProjectEulerStatements.P9.naive
  split
  · exact Finset.le_max' _ x hx
  · rename_i hne
    exact False.elim (hne ⟨x, hx⟩)

lemma candidate_le_naive (total m n : Nat) :
    candidate total m n ≤ ProjectEulerStatements.P9.naive total := by
  by_cases hzero : candidate total m n = 0
  · rw [hzero]
    exact Nat.zero_le _
  · exact mem_tripletProducts_le_naive (candidate_mem_tripletProducts_of_ne_zero hzero)

lemma foldl_max_candidate_le_naive (total m : Nat) (l : List Nat) {best : Nat}
    (hbest : best ≤ ProjectEulerStatements.P9.naive total) :
    l.foldl (fun acc x => Nat.max acc (candidate total m x)) best ≤
      ProjectEulerStatements.P9.naive total := by
  induction l generalizing best with
  | nil => simpa using hbest
  | cons x xs ih =>
      rw [List.foldl_cons]
      apply ih
      exact max_le hbest (candidate_le_naive total m x)

lemma loopN_le_naive (total m n best : Nat)
    (hbest : best ≤ ProjectEulerStatements.P9.naive total) :
    loopN total m n best ≤ ProjectEulerStatements.P9.naive total := by
  rw [loopN_eq_fold]
  exact foldl_max_candidate_le_naive total m (List.range' n (m - n)) hbest

lemma foldl_loopM_step_le_naive (total : Nat) (l : List Nat) {best : Nat}
    (hbest : best ≤ ProjectEulerStatements.P9.naive total) :
    l.foldl (fun acc x =>
      if 2 * x * (x + 1) > total then acc else Nat.max acc (loopN total x 1 0)) best ≤
        ProjectEulerStatements.P9.naive total := by
  induction l generalizing best with
  | nil => simpa using hbest
  | cons x xs ih =>
      rw [List.foldl_cons]
      by_cases hstop : 2 * x * (x + 1) > total
      · simp [hstop]
        exact ih hbest
      · simp [hstop]
        apply ih
        exact max_le hbest (loopN_le_naive total x 1 0 (Nat.zero_le _))

lemma loopM_le_naive (total m best : Nat)
    (hbest : best ≤ ProjectEulerStatements.P9.naive total) :
    loopM total m best ≤ ProjectEulerStatements.P9.naive total := by
  rw [loopM_eq_fold]
  exact foldl_loopM_step_le_naive total (List.range' m (total + 1 - m)) hbest

lemma euclidSearch_le_naive (total : Nat) :
    loopM total 2 0 ≤ ProjectEulerStatements.P9.naive total := by
  exact loopM_le_naive total 2 0 (Nat.zero_le _)

lemma exists_ordered_triple_of_mem_tripletProducts {total x : Nat}
    (hx : x ∈ ProjectEulerStatements.P9.tripletProducts total) :
    ∃ a b c : Nat,
      1 ≤ a ∧ a ≤ total ∧ 1 ≤ b ∧ b ≤ total ∧ c = total - a - b ∧
        a < b ∧ b < c ∧ a ^ 2 + b ^ 2 = c ^ 2 ∧ x = a * b * c := by
  unfold ProjectEulerStatements.P9.tripletProducts at hx
  rw [Finset.mem_image] at hx
  rcases hx with ⟨p, hp, rfl⟩
  rw [Finset.mem_filter] at hp
  rcases hp with ⟨hp_range, hp_pred⟩
  rcases p with ⟨a, b⟩
  simp at hp_range hp_pred
  refine ⟨a, b, total - a - b, ?_, ?_, ?_, ?_, rfl, ?_, ?_, ?_, rfl⟩
  · exact hp_range.1.1
  · exact hp_range.1.2
  · exact hp_range.2.1
  · exact hp_range.2.2
  · exact hp_pred.1
  · exact hp_pred.2.1
  · exact hp_pred.2.2

lemma int_classification_of_nat_pythagorean {a b c : Nat}
    (hpyt : a ^ 2 + b ^ 2 = c ^ 2) :
    ∃ k m n : Int,
      (((a : Int) = k * (m ^ 2 - n ^ 2) ∧ (b : Int) = k * (2 * m * n)) ∨
        ((a : Int) = k * (2 * m * n) ∧ (b : Int) = k * (m ^ 2 - n ^ 2))) ∧
      ((c : Int) = k * (m ^ 2 + n ^ 2) ∨ (c : Int) = -k * (m ^ 2 + n ^ 2)) := by
  have hpt : PythagoreanTriple (a : Int) (b : Int) (c : Int) := by
    unfold PythagoreanTriple
    norm_num [pow_two] at hpyt ⊢
    exact_mod_cast hpyt
  exact PythagoreanTriple.classification.mp hpt

lemma directTripletProducts_eq_statement (total : Nat) :
    directTripletProducts total = ProjectEulerStatements.P9.tripletProducts total := by
  rfl

lemma directSolve_eq_naive (total : Nat) :
    directSolve total = ProjectEulerStatements.P9.naive total := by
  unfold directSolve ProjectEulerStatements.P9.naive
  rw [directTripletProducts_eq_statement]

theorem equiv (n : Nat) : ProjectEulerStatements.P9.naive n = solve n := by
  unfold solve
  rw [directSolve_eq_naive]
  exact (Nat.max_eq_right (euclidSearch_le_naive n)).symm
end ProjectEulerSolutions.P9
