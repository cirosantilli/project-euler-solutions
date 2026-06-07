import ProjectEulerSolutions.P9
import Mathlib.NumberTheory.PythagoreanTriples
import Mathlib.Tactic
namespace ProjectEulerSolutions.P9

def euclidS (m n : Nat) : Nat :=
  2 * m * (m + n)

def euclidK (total m n : Nat) : Nat :=
  total / euclidS m n

def euclidA (total m n : Nat) : Nat :=
  euclidK total m n * (m * m - n * n)

def euclidB (total m n : Nat) : Nat :=
  euclidK total m n * (2 * m * n)

def euclidC (total m n : Nat) : Nat :=
  euclidK total m n * (m * m + n * n)

def candidateValid (total m n : Nat) : Prop :=
  let a := euclidA total m n
  let b := euclidB total m n
  let c := euclidC total m n
  total % euclidS m n = 0 ∧
    a > 0 ∧ ((a < b ∧ b < c) ∨ (b < a ∧ a < c)) ∧
      a * a + b * b = c * c ∧ a + b + c = total

instance candidateValidDecidable (total m n : Nat) : Decidable (candidateValid total m n) := by
  unfold candidateValid
  infer_instance

def candidate (total m n : Nat) : Nat :=
  if candidateValid total m n then
    euclidA total m n * euclidB total m n * euclidC total m n
  else
    0

lemma loopN_eq_foldr (total m n : Nat) :
    loopN total m n =
      (List.range' n (m - n)).foldr (fun x acc => Nat.max (candidate total m x) acc) 0 := by
  induction n using loopN.induct total m with
  | case1 n h =>
      rw [loopN.eq_def]
      have hsub : m - n = 0 := Nat.sub_eq_zero_of_le h
      simp [h, hsub]
  | case2 n h ih =>
      rw [loopN.eq_def]
      simp [h]
      rw [ih]
      have hmn : m - n = (m - (n + 1)) + 1 := by omega
      rw [hmn, List.range'_succ]
      simp [candidate, candidateValid, euclidA, euclidB, euclidC, euclidK, euclidS]

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

lemma foldr_if_pos_eq_zero (total : Nat) (l : List Nat)
    (h : ∀ x ∈ l, 2 * x * (x + 1) > total) :
    l.foldr (fun x acc =>
      if 2 * x * (x + 1) > total then acc else Nat.max (loopN total x 1) acc) 0 = 0 := by
  induction l with
  | nil => simp
  | cons x xs ih =>
      have hx : 2 * x * (x + 1) > total := h x (by simp)
      have hxs : ∀ y ∈ xs, 2 * y * (y + 1) > total := by
        intro y hy
        exact h y (by simp [hy])
      simp [hx, ih hxs]

lemma loopM_eq_foldr (total m : Nat) :
    loopM total m =
      (List.range' m (total + 1 - m)).foldr
        (fun x acc =>
          if 2 * x * (x + 1) > total then acc else Nat.max (loopN total x 1) acc) 0 := by
  induction m using loopM.induct total with
  | case1 m hstop =>
      rw [loopM.eq_def]
      simp [hstop]
      exact (foldr_if_pos_eq_zero total (List.range' m (total + 1 - m)) (by
        intro x hx
        exact loopM_stop_mono hstop (range'_ge_start hx))).symm
  | case2 m hstop ih =>
      rw [loopM.eq_def]
      simp [hstop]
      rw [ih]
      have hm_le : m ≤ total := by
        have hmul : m ≤ 2 * m * (m + 1) := by nlinarith
        exact le_trans hmul (le_of_not_gt hstop)
      have hlen : total + 1 - m = (total + 1 - (m + 1)) + 1 := by omega
      rw [hlen, List.range'_succ]
      simp [hstop]

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
    rcases hv with ⟨_hmod, ha0, horder, hpyt, hsum⟩
    have hpyt_pow : a ^ 2 + b ^ 2 = c ^ 2 := by
      simpa [pow_two, a, b, c] using hpyt
    have hpyt_swap_pow : b ^ 2 + a ^ 2 = c ^ 2 := by
      rw [Nat.add_comm]
      exact hpyt_pow
    have hsum' : a + b + c = total := by simpa [a, b, c] using hsum
    unfold ProjectEulerStatements.P9.tripletProducts
    rcases horder with ⟨hab, hbc⟩ | ⟨hba, hac⟩
    · apply Finset.mem_image.mpr
      refine ⟨(a, b), ?_, ?_⟩
      · rw [Finset.mem_filter]
        constructor
        · simp
          omega
        · change a < b ∧ b < total - a - b ∧ a ^ 2 + b ^ 2 = (total - a - b) ^ 2
          rw [show total - a - b = c by omega]
          exact ⟨hab, hbc, hpyt_pow⟩
      · change a * b * (total - a - b) = a * b * c
        rw [show total - a - b = c by omega]
    · apply Finset.mem_image.mpr
      refine ⟨(b, a), ?_, ?_⟩
      · rw [Finset.mem_filter]
        have hb0 : 0 < b := by
          by_contra hb
          have hbz : b = 0 := by omega
          nlinarith [hpyt, hac]
        constructor
        · simp
          omega
        · change b < a ∧ a < total - b - a ∧ b ^ 2 + a ^ 2 = (total - b - a) ^ 2
          rw [show total - b - a = c by omega]
          exact ⟨hba, hac, hpyt_swap_pow⟩
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

lemma foldr_max_candidate_le_naive (total m : Nat) (l : List Nat) :
    l.foldr (fun x acc => Nat.max (candidate total m x) acc) 0 ≤
      ProjectEulerStatements.P9.naive total := by
  induction l with
  | nil => simp
  | cons x xs ih =>
      simp
      exact ⟨candidate_le_naive total m x, ih⟩

lemma loopN_le_naive (total m n : Nat) :
    loopN total m n ≤ ProjectEulerStatements.P9.naive total := by
  rw [loopN_eq_foldr]
  exact foldr_max_candidate_le_naive total m (List.range' n (m - n))

lemma foldr_loopM_step_le_naive (total : Nat) (l : List Nat) :
    l.foldr (fun x acc =>
      if 2 * x * (x + 1) > total then acc else Nat.max (loopN total x 1) acc) 0 ≤
        ProjectEulerStatements.P9.naive total := by
  induction l with
  | nil => simp
  | cons x xs ih =>
      by_cases hstop : 2 * x * (x + 1) > total
      · simp [hstop, ih]
      · simp [hstop]
        exact ⟨loopN_le_naive total x 1, ih⟩

lemma loopM_le_naive (total m : Nat) :
    loopM total m ≤ ProjectEulerStatements.P9.naive total := by
  rw [loopM_eq_foldr]
  exact foldr_loopM_step_le_naive total (List.range' m (total + 1 - m))

lemma solve_le_naive (total : Nat) :
    solve total ≤ ProjectEulerStatements.P9.naive total := by
  unfold solve
  exact loopM_le_naive total 2

lemma le_foldr_max_of_mem {l : List Nat} {f : Nat → Nat} {x : Nat}
    (hx : x ∈ l) : f x ≤ l.foldr (fun y acc => Nat.max (f y) acc) 0 := by
  induction l with
  | nil => simp at hx
  | cons y ys ih =>
      simp at hx
      change f x ≤ Nat.max (f y) (ys.foldr (fun y acc => Nat.max (f y) acc) 0)
      rcases hx with rfl | hx
      · exact Nat.le_max_left _ _
      · exact le_trans (ih hx) (Nat.le_max_right _ _)

lemma candidate_le_loopN_of_mem {total m n : Nat}
    (hn : n ∈ List.range' 1 (m - 1)) :
    candidate total m n ≤ loopN total m 1 := by
  rw [loopN_eq_foldr]
  exact le_foldr_max_of_mem hn

lemma loopN_le_foldr_loopM_step_of_mem {total m : Nat} {l : List Nat}
    (hm : m ∈ l) (hstop : ¬2 * m * (m + 1) > total) :
    loopN total m 1 ≤
      l.foldr
        (fun x acc =>
          if 2 * x * (x + 1) > total then acc else Nat.max (loopN total x 1) acc) 0 := by
  induction l with
  | nil => simp at hm
  | cons x xs ih =>
      simp at hm
      by_cases hx : x = m
      · subst hx
        simp [hstop]
      · have hm_xs : m ∈ xs := by
          rcases hm with hxm | hxs
          · exact False.elim (hx hxm.symm)
          · exact hxs
        by_cases hxstop : 2 * x * (x + 1) > total
        · simp [hxstop]
          exact ih hm_xs
        · change loopN total m 1 ≤
            (if 2 * x * (x + 1) > total then
              xs.foldr
                (fun x acc =>
                  if 2 * x * (x + 1) > total then acc else Nat.max (loopN total x 1) acc) 0
            else
              Nat.max (loopN total x 1)
                (xs.foldr
                  (fun x acc =>
                    if 2 * x * (x + 1) > total then acc else Nat.max (loopN total x 1) acc) 0))
          rw [if_neg hxstop]
          exact le_trans (ih hm_xs) (Nat.le_max_right _ _)

lemma loopN_le_loopM_of_mem {total m : Nat}
    (hm : m ∈ List.range' 2 (total + 1 - 2)) (hstop : ¬2 * m * (m + 1) > total) :
    loopN total m 1 ≤ loopM total 2 := by
  rw [loopM_eq_foldr]
  exact loopN_le_foldr_loopM_step_of_mem hm hstop

lemma candidate_le_solve_of_bounds {total m n : Nat}
    (hm : m ∈ List.range' 2 (total + 1 - 2))
    (hn : n ∈ List.range' 1 (m - 1))
    (hstop : ¬2 * m * (m + 1) > total) :
    candidate total m n ≤ solve total := by
  unfold solve
  exact le_trans (candidate_le_loopN_of_mem hn) (loopN_le_loopM_of_mem hm hstop)

lemma natAbs_sq_sub_of_le (m n : Int) (h : n.natAbs ≤ m.natAbs) :
    (m ^ 2 - n ^ 2).natAbs = m.natAbs ^ 2 - n.natAbs ^ 2 := by
  have hnonneg : 0 ≤ m ^ 2 - n ^ 2 := by
    have habs : |n| ≤ |m| := by
      rw [← Int.natCast_natAbs n, ← Int.natCast_natAbs m]
      exact_mod_cast h
    have hsquare : n ^ 2 ≤ m ^ 2 := by
      rwa [sq_le_sq]
    nlinarith
  apply Nat.cast_injective (R := Int)
  rw [Int.natAbs_of_nonneg hnonneg]
  rw [Nat.cast_sub]
  · norm_num [Int.natCast_natAbs, sq_abs]
  · exact pow_le_pow_left₀ (Nat.zero_le _) h 2

lemma natAbs_sq_sub_eq_max_min (m n : Int) :
    (m ^ 2 - n ^ 2).natAbs =
      (max m.natAbs n.natAbs) ^ 2 - (min m.natAbs n.natAbs) ^ 2 := by
  by_cases h : n.natAbs ≤ m.natAbs
  · rw [max_eq_left h, min_eq_right h]
    exact natAbs_sq_sub_of_le m n h
  · have h' : m.natAbs ≤ n.natAbs := by omega
    rw [max_eq_right h', min_eq_left h']
    rw [show m ^ 2 - n ^ 2 = -(n ^ 2 - m ^ 2) by ring]
    rw [Int.natAbs_neg]
    exact natAbs_sq_sub_of_le n m h'

lemma natAbs_sq_add_eq_max_min (m n : Int) :
    (m ^ 2 + n ^ 2).natAbs =
      (max m.natAbs n.natAbs) ^ 2 + (min m.natAbs n.natAbs) ^ 2 := by
  have hnonneg : 0 ≤ m ^ 2 + n ^ 2 := by positivity
  apply Nat.cast_injective (R := Int)
  rw [Int.natAbs_of_nonneg hnonneg]
  by_cases h : n.natAbs ≤ m.natAbs
  · rw [max_eq_left h, min_eq_right h]
    norm_num [Int.natCast_natAbs, sq_abs]
  · have h' : m.natAbs ≤ n.natAbs := by omega
    rw [max_eq_right h', min_eq_left h']
    norm_num [Int.natCast_natAbs, sq_abs]
    ring

lemma natAbs_two_mul_eq_max_min (m n : Int) :
    (2 * m * n).natAbs = 2 * (max m.natAbs n.natAbs) * (min m.natAbs n.natAbs) := by
  rw [Int.natAbs_mul, Int.natAbs_mul]
  by_cases h : n.natAbs ≤ m.natAbs
  · rw [max_eq_left h, min_eq_right h]
    simp
  · have h' : m.natAbs ≤ n.natAbs := by omega
    rw [max_eq_right h', min_eq_left h']
    simp [Nat.mul_comm, Nat.mul_left_comm]

lemma euclid_sum_identity (K M N : Nat) (hNM : N ≤ M) :
    K * (M ^ 2 - N ^ 2) + K * (2 * M * N) + K * (M ^ 2 + N ^ 2) =
      K * euclidS M N := by
  have hN2leM2 : N ^ 2 ≤ M ^ 2 := pow_le_pow_left₀ (Nat.zero_le _) hNM 2
  have hsub : M ^ 2 - N ^ 2 + N ^ 2 = M ^ 2 := Nat.sub_add_cancel hN2leM2
  unfold euclidS
  nlinarith [hsub]

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

lemma nat_euclid_of_int_class {a b c : Nat} (ha0 : 0 < a) (hb0 : 0 < b) (_hc0 : 0 < c)
    {k m n : Int}
    (hlegs : (((a : Int) = k * (m ^ 2 - n ^ 2) ∧ (b : Int) = k * (2 * m * n)) ∨
        ((a : Int) = k * (2 * m * n) ∧ (b : Int) = k * (m ^ 2 - n ^ 2))))
    (hc : ((c : Int) = k * (m ^ 2 + n ^ 2) ∨ (c : Int) = -k * (m ^ 2 + n ^ 2))) :
    ∃ K M N : Nat,
      0 < K ∧ 0 < N ∧ N < M ∧
      ((((a = K * (M ^ 2 - N ^ 2) ∧ b = K * (2 * M * N)) ∨
        (a = K * (2 * M * N) ∧ b = K * (M ^ 2 - N ^ 2))) ∧
        c = K * (M ^ 2 + N ^ 2))) := by
  let K := k.natAbs
  let M := max m.natAbs n.natAbs
  let N := min m.natAbs n.natAbs
  have hdiff : (m ^ 2 - n ^ 2).natAbs = M ^ 2 - N ^ 2 := by
    simpa [M, N] using natAbs_sq_sub_eq_max_min m n
  have hsum : (m ^ 2 + n ^ 2).natAbs = M ^ 2 + N ^ 2 := by
    simpa [M, N] using natAbs_sq_add_eq_max_min m n
  have htwo : (2 * m * n).natAbs = 2 * M * N := by
    simpa [M, N] using natAbs_two_mul_eq_max_min m n
  have hc_nat : c = K * (M ^ 2 + N ^ 2) := by
    rcases hc with hc | hc
    · have hcabs : (k * (m ^ 2 + n ^ 2)).natAbs = c := by
        rw [← hc]
        simp
      rw [← hcabs]
      rw [Int.natAbs_mul, hsum]
    · have hcabs : (k * (m ^ 2 + n ^ 2)).natAbs = c := by
        rw [show (k * (m ^ 2 + n ^ 2)).natAbs = (-(k * (m ^ 2 + n ^ 2))).natAbs by
          rw [Int.natAbs_neg]]
        rw [show -(k * (m ^ 2 + n ^ 2)) = (c : Int) by rw [hc]; ring]
        simp
      rw [← hcabs]
      rw [Int.natAbs_mul, hsum]
  rcases hlegs with ⟨ha, hb⟩ | ⟨ha, hb⟩
  · have ha_nat : a = K * (M ^ 2 - N ^ 2) := by
      have haabs : (k * (m ^ 2 - n ^ 2)).natAbs = a := by
        rw [← ha]
        simp
      rw [← haabs]
      rw [Int.natAbs_mul, hdiff]
    have hb_nat : b = K * (2 * M * N) := by
      have hbabs : (k * (2 * m * n)).natAbs = b := by
        rw [← hb]
        simp
      rw [← hbabs]
      rw [Int.natAbs_mul, htwo]
    have hK : 0 < K := by nlinarith [ha0, ha_nat]
    have hN : 0 < N := by
      by_contra hN
      have hNz : N = 0 := by omega
      simp [hNz] at hb_nat
      omega
    have hNM : N < M := by
      have hdiffpos : 0 < M ^ 2 - N ^ 2 := by nlinarith [ha0, ha_nat]
      by_contra hnot
      have hMN : M ≤ N := by omega
      have hpow : M ^ 2 ≤ N ^ 2 := pow_le_pow_left₀ (Nat.zero_le _) hMN 2
      omega
    exact ⟨K, M, N, hK, hN, hNM, Or.inl ⟨ha_nat, hb_nat⟩, hc_nat⟩
  · have ha_nat : a = K * (2 * M * N) := by
      have haabs : (k * (2 * m * n)).natAbs = a := by
        rw [← ha]
        simp
      rw [← haabs]
      rw [Int.natAbs_mul, htwo]
    have hb_nat : b = K * (M ^ 2 - N ^ 2) := by
      have hbabs : (k * (m ^ 2 - n ^ 2)).natAbs = b := by
        rw [← hb]
        simp
      rw [← hbabs]
      rw [Int.natAbs_mul, hdiff]
    have hK : 0 < K := by nlinarith [hb0, hb_nat]
    have hN : 0 < N := by
      by_contra hN
      have hNz : N = 0 := by omega
      simp [hNz] at ha_nat
      omega
    have hNM : N < M := by
      have hdiffpos : 0 < M ^ 2 - N ^ 2 := by nlinarith [hb0, hb_nat]
      by_contra hnot
      have hMN : M ≤ N := by omega
      have hpow : M ^ 2 ≤ N ^ 2 := pow_le_pow_left₀ (Nat.zero_le _) hMN 2
      omega
    exact ⟨K, M, N, hK, hN, hNM, Or.inr ⟨ha_nat, hb_nat⟩, hc_nat⟩

lemma exists_candidate_of_tripletProducts {total x : Nat}
    (hx : x ∈ ProjectEulerStatements.P9.tripletProducts total) :
    ∃ M N : Nat,
      M ∈ List.range' 2 (total + 1 - 2) ∧
      N ∈ List.range' 1 (M - 1) ∧
      ¬2 * M * (M + 1) > total ∧
      candidate total M N = x := by
  obtain ⟨a, b, c, ha1, _hat, hb1, _hbt, hcdef, hab, hbc, hpyt, rfl⟩ :=
    exists_ordered_triple_of_mem_tripletProducts hx
  have ha0 : 0 < a := by omega
  have hb0 : 0 < b := by omega
  have hc0 : 0 < c := by omega
  have hsumabc : a + b + c = total := by omega
  obtain ⟨k, m, n, hlegs, hc⟩ := int_classification_of_nat_pythagorean hpyt
  obtain ⟨K, M, N, hK, hN, hNM, hlegsnat, hcnat⟩ :=
    nat_euclid_of_int_class ha0 hb0 hc0 hlegs hc
  have hSpos : 0 < euclidS M N := by
    unfold euclidS
    nlinarith [hN, hNM]
  have hS1leS : 2 * M * (M + 1) ≤ euclidS M N := by
    unfold euclidS
    nlinarith [hN]
  have hMleS : M ≤ euclidS M N := by
    unfold euclidS
    nlinarith [hN, hNM]
  rcases hlegsnat with ⟨hae, hbe⟩ | ⟨hae, hbe⟩
  · have htotal : total = K * euclidS M N := by
      rw [← hsumabc, hae, hbe, hcnat]
      exact euclid_sum_identity K M N (le_of_lt hNM)
    have hSle_total : euclidS M N ≤ total := by
      rw [htotal]
      exact Nat.le_mul_of_pos_left (euclidS M N) hK
    have hMle_total : M ≤ total := le_trans hMleS hSle_total
    have hKdiv : euclidK total M N = K := by
      unfold euclidK
      rw [htotal]
      rw [show K * euclidS M N = euclidS M N * K by ring]
      exact Nat.mul_div_right K hSpos
    have hA : euclidA total M N = a := by
      unfold euclidA
      rw [hKdiv]
      simpa [pow_two] using hae.symm
    have hB : euclidB total M N = b := by
      unfold euclidB
      rw [hKdiv]
      simpa [pow_two] using hbe.symm
    have hC : euclidC total M N = c := by
      unfold euclidC
      rw [hKdiv]
      simpa [pow_two] using hcnat.symm
    have hvalid : candidateValid total M N := by
      unfold candidateValid
      refine ⟨?_, ?_, ?_, ?_, ?_⟩
      · rw [htotal]
        rw [show K * euclidS M N = euclidS M N * K by ring]
        exact Nat.mul_mod_right (euclidS M N) K
      · rw [hA]
        exact ha0
      · left
        rw [hA, hB, hC]
        exact ⟨hab, hbc⟩
      · rw [hA, hB, hC]
        simpa [pow_two] using hpyt
      · rw [hA, hB, hC]
        exact hsumabc
    refine ⟨M, N, ?_, ?_, ?_, ?_⟩
    · rw [List.mem_range'_1]
      omega
    · rw [List.mem_range'_1]
      omega
    · omega
    · unfold candidate
      simp [hvalid]
      rw [hA, hB, hC]
  · have htotal : total = K * euclidS M N := by
      rw [← hsumabc, hae, hbe, hcnat]
      rw [Nat.add_comm (K * (2 * M * N))]
      exact euclid_sum_identity K M N (le_of_lt hNM)
    have hSle_total : euclidS M N ≤ total := by
      rw [htotal]
      exact Nat.le_mul_of_pos_left (euclidS M N) hK
    have hMle_total : M ≤ total := le_trans hMleS hSle_total
    have hKdiv : euclidK total M N = K := by
      unfold euclidK
      rw [htotal]
      rw [show K * euclidS M N = euclidS M N * K by ring]
      exact Nat.mul_div_right K hSpos
    have hA : euclidA total M N = b := by
      unfold euclidA
      rw [hKdiv]
      simpa [pow_two] using hbe.symm
    have hB : euclidB total M N = a := by
      unfold euclidB
      rw [hKdiv]
      simpa [pow_two] using hae.symm
    have hC : euclidC total M N = c := by
      unfold euclidC
      rw [hKdiv]
      simpa [pow_two] using hcnat.symm
    have hvalid : candidateValid total M N := by
      unfold candidateValid
      refine ⟨?_, ?_, ?_, ?_, ?_⟩
      · rw [htotal]
        rw [show K * euclidS M N = euclidS M N * K by ring]
        exact Nat.mul_mod_right (euclidS M N) K
      · rw [hA]
        exact hb0
      · right
        rw [hA, hB, hC]
        exact ⟨hab, hbc⟩
      · rw [hA, hB, hC]
        rw [Nat.add_comm]
        simpa [pow_two] using hpyt
      · rw [hA, hB, hC]
        omega
    refine ⟨M, N, ?_, ?_, ?_, ?_⟩
    · rw [List.mem_range'_1]
      omega
    · rw [List.mem_range'_1]
      omega
    · omega
    · unfold candidate
      simp [hvalid]
      rw [hA, hB, hC]
      ring

lemma naive_le_solve (total : Nat) :
    ProjectEulerStatements.P9.naive total ≤ solve total := by
  unfold ProjectEulerStatements.P9.naive
  split
  · rename_i hne
    let x := (ProjectEulerStatements.P9.tripletProducts total).max' hne
    have hx : x ∈ ProjectEulerStatements.P9.tripletProducts total := Finset.max'_mem _ _
    obtain ⟨M, N, hm, hn, hstop, hcandidate⟩ := exists_candidate_of_tripletProducts hx
    change x ≤ solve total
    rw [← hcandidate]
    exact candidate_le_solve_of_bounds hm hn hstop
  · exact Nat.zero_le _

theorem equiv (n : Nat) : ProjectEulerStatements.P9.naive n = solve n := by
  exact le_antisymm (naive_le_solve n) (solve_le_naive n)
end ProjectEulerSolutions.P9
