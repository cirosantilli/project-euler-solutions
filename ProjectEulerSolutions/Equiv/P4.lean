import ProjectEulerSolutions.P4
import Batteries.Data.Nat.Digits
import Batteries.Data.String.Lemmas
import Mathlib.Tactic

namespace ProjectEulerSolutions.P4

namespace Equiv

lemma toDigits_eq_digitChars (n : Nat) (hn : 0 < n) :
    Nat.toDigits 10 n = (Nat.digits 10 n).reverse.map Nat.digitChar := by
  induction n using Nat.strong_induction_on with
  | h n ih =>
      by_cases hlt : n < 10
      · rw [Nat.toDigits_of_lt_base hlt, Nat.digits_of_lt 10 n (Nat.ne_of_gt hn) hlt]
        simp
      · have hn10 : 10 ≤ n := le_of_not_gt hlt
        have hdivpos : 0 < n / 10 := Nat.div_pos hn10 (by norm_num)
        have hdivlt : n / 10 < n := Nat.div_lt_self hn (by norm_num)
        have hmodlt : n % 10 < 10 := Nat.mod_lt n (by norm_num)
        have hn_eq : n = 10 * (n / 10) + n % 10 := by
          conv_lhs => rw [← Nat.div_add_mod n 10]
        calc
          Nat.toDigits 10 n = Nat.toDigits 10 (10 * (n / 10) + n % 10) := by
            exact congrArg (Nat.toDigits 10) hn_eq
          _ = Nat.toDigits 10 (n / 10) ++ Nat.toDigits 10 (n % 10) := by
            exact (Nat.toDigits_append_toDigits (b := 10) (n := n / 10) (d := n % 10)
              (by norm_num) hdivpos hmodlt).symm
          _ = (Nat.digits 10 (n / 10)).reverse.map Nat.digitChar ++
                [Nat.digitChar (n % 10)] := by
            rw [ih (n / 10) hdivlt hdivpos, Nat.toDigits_of_lt_base hmodlt]
          _ = (Nat.digits 10 n).reverse.map Nat.digitChar := by
            rw [Nat.digits_of_two_le_of_pos (b := 10) (n := n) (by norm_num) hn]
            simp [List.map_append]

lemma digitChar_inj_of_lt {a b : Nat} (ha : a < 10) (hb : b < 10)
    (h : Nat.digitChar a = Nat.digitChar b) : a = b := by
  interval_cases a <;> interval_cases b <;> simp [Nat.digitChar] at h ⊢

lemma map_digitChar_inj_of_lt {xs ys : List Nat} (hxs : ∀ d ∈ xs, d < 10)
    (hys : ∀ d ∈ ys, d < 10) (h : xs.map Nat.digitChar = ys.map Nat.digitChar) :
    xs = ys := by
  induction xs generalizing ys with
  | nil =>
      cases ys with
      | nil => rfl
      | cons y ys =>
          simp at h
  | cons x xs ih =>
      cases ys with
      | nil =>
          simp at h
      | cons y ys =>
          simp at h
          have hxy : x = y := digitChar_inj_of_lt (hxs x (by simp)) (hys y (by simp)) h.1
          subst y
          have hxs_tail : ∀ d ∈ xs, d < 10 := by
            intro d hd
            exact hxs d (by simp [hd])
          have hys_tail : ∀ d ∈ ys, d < 10 := by
            intro d hd
            exact hys d (by simp [hd])
          rw [ih hxs_tail hys_tail h.2]

lemma palindrome_bool_iff (n : Nat) :
    isPalindrome n = true ↔ ProjectEulerStatements.P4.isPalindrome n := by
  by_cases hn : n = 0
  · subst n
    native_decide
  · have hnpos : 0 < n := Nat.pos_of_ne_zero hn
    have hdigits : ∀ d ∈ Nat.digits 10 n, d < 10 := by
      intro d hd
      exact Nat.digits_lt_base (b := 10) (m := n) (d := d) (by norm_num) hd
    have hdigits_rev : ∀ d ∈ (Nat.digits 10 n).reverse, d < 10 := by
      intro d hd
      exact hdigits d (by simpa using hd)
    have hdata : (toString n).data = (Nat.digits 10 n).reverse.map Nat.digitChar := by
      rw [show (toString n).data = Nat.toDigits 10 n by rfl]
      exact toDigits_eq_digitChars n hnpos
    constructor
    · intro h
      unfold isPalindrome at h
      simp only [decide_eq_true_eq] at h
      have hdata_pal := congrArg String.data h
      rw [hdata] at hdata_pal
      have hmap : (Nat.digits 10 n).reverse.map Nat.digitChar =
          (Nat.digits 10 n).map Nat.digitChar := by
        simpa [List.map_reverse] using hdata_pal
      have hrev : (Nat.digits 10 n).reverse = Nat.digits 10 n :=
        map_digitChar_inj_of_lt hdigits_rev hdigits hmap
      simpa [ProjectEulerStatements.P4.isPalindrome] using hrev.symm
    · intro h
      unfold isPalindrome
      simp only [decide_eq_true_eq]
      apply String.ext
      rw [hdata]
      have hrev : (Nat.digits 10 n).reverse = Nat.digits 10 n := by
        simpa [ProjectEulerStatements.P4.isPalindrome] using h.symm
      have hmap : (Nat.digits 10 n).reverse.map Nat.digitChar =
          (Nat.digits 10 n).map Nat.digitChar := by
        rw [hrev]
      simpa [hdata, List.map_reverse] using hmap

lemma loopB_ge_best (a b lo best bestA bestB : Nat) :
    best ≤ (loopB a b lo best bestA bestB).1 := by
  induction b using loopB.induct a lo best with
  | case1 b hblo =>
      rw [loopB.eq_1]
      simp [hblo]
  | case2 hblo =>
      rw [loopB.eq_1]
      simp [hblo]
  | case3 b hblo hb0 prod hle =>
      change a * b ≤ best at hle
      rw [loopB.eq_1]
      simp [hblo, hb0, hle]
  | case4 b hblo hb0 prod hnot hpal =>
      change ¬a * b ≤ best at hnot
      change isPalindrome (a * b) = true at hpal
      rw [loopB.eq_1]
      simp [hblo, hb0, hnot, hpal]
      omega
  | case5 b hblo hb0 prod hnot hpal ih =>
      change ¬a * b ≤ best at hnot
      change ¬isPalindrome (a * b) = true at hpal
      rw [loopB.eq_1]
      simp [hblo, hb0, hnot, hpal]
      exact ih

lemma loopB_covers (a b lo best bestA bestB c : Nat)
    (hlo : lo ≤ c) (hcb : c ≤ b) (hpal : ProjectEulerStatements.P4.isPalindrome (a * c)) :
    a * c ≤ (loopB a b lo best bestA bestB).1 := by
  induction b using loopB.induct a lo best with
  | case1 b hblo =>
      omega
  | case2 hblo =>
      rw [loopB.eq_1]
      simp [hblo]
      have hc : c = 0 := by omega
      subst c
      simp
  | case3 b hblo hb0 prod hle =>
      change a * b ≤ best at hle
      rw [loopB.eq_1]
      simp [hblo, hb0, hle]
      nlinarith [Nat.mul_le_mul_left a hcb]
  | case4 b hblo hb0 prod hnot hpalb =>
      change ¬a * b ≤ best at hnot
      change isPalindrome (a * b) = true at hpalb
      rw [loopB.eq_1]
      simp [hblo, hb0, hnot, hpalb]
      exact Nat.mul_le_mul_left a hcb
  | case5 b hblo hb0 prod hnot hpalb ih =>
      change ¬a * b ≤ best at hnot
      change ¬isPalindrome (a * b) = true at hpalb
      rw [loopB.eq_1]
      simp [hblo, hb0, hnot, hpalb]
      by_cases hcb' : c ≤ b - 1
      · exact ih hcb'
      · have hc : c = b := by omega
        subst c
        have hbpal : isPalindrome (a * b) = true := (palindrome_bool_iff (a * b)).2 hpal
        contradiction

lemma loopB_candidate (a b lo best bestA bestB : Nat) :
    (loopB a b lo best bestA bestB).1 = best ∨
      ∃ c, lo ≤ c ∧ c ≤ b ∧ ProjectEulerStatements.P4.isPalindrome (a * c) ∧
        (loopB a b lo best bestA bestB).1 = a * c := by
  induction b using loopB.induct a lo best with
  | case1 b hblo =>
      rw [loopB.eq_1]
      simp [hblo]
  | case2 hblo =>
      rw [loopB.eq_1]
      simp [hblo]
  | case3 b hblo hb0 prod hle =>
      change a * b ≤ best at hle
      rw [loopB.eq_1]
      simp [hblo, hb0, hle]
  | case4 b hblo hb0 prod hnot hpal =>
      change ¬a * b ≤ best at hnot
      change isPalindrome (a * b) = true at hpal
      right
      refine ⟨b, by omega, le_rfl, ?_, ?_⟩
      · exact (palindrome_bool_iff (a * b)).1 hpal
      · rw [loopB.eq_1]
        simp [hblo, hb0, hnot, hpal]
  | case5 b hblo hb0 prod hnot hpal ih =>
      change ¬a * b ≤ best at hnot
      change ¬isPalindrome (a * b) = true at hpal
      rw [loopB.eq_1]
      simp [hblo, hb0, hnot, hpal]
      rcases ih with hbest | hcand
      · exact Or.inl hbest
      · rcases hcand with ⟨c, hlo, hcb, hpalc, hres⟩
        exact Or.inr ⟨c, hlo, by omega, hpalc, hres⟩

lemma loopA_ge_best (a lo hi best bestA bestB : Nat) :
    best ≤ (loopA a lo hi best bestA bestB).1 := by
  induction a, best, bestA, bestB using loopA.induct lo hi with
  | case1 a best bestA bestB ha =>
      rw [loopA.eq_1]
      simp [ha]
  | case2 a best bestA bestB ha hbest =>
      rw [loopA.eq_1]
      simp [ha, hbest]
  | case3 best bestA bestB best' bestA' bestB' ha hbest hloop =>
      have hbest0 : best = 0 := by omega
      subst best
      exact Nat.zero_le _
  | case4 a best bestA bestB ha hbest best' bestA' bestB' hloop hzero ih =>
      rw [loopA.eq_1]
      simp [ha, hbest, hloop, hzero]
      have hB : best ≤ best' := by
        have hB0 := loopB_ge_best a a lo best bestA bestB
        rw [hloop] at hB0
        exact hB0
      exact le_trans hB ih

lemma loopA_covers (a lo hi best bestA bestB x y : Nat) (hahi : a ≤ hi)
    (hlox : lo ≤ x) (hxa : x ≤ a) (hloy : lo ≤ y) (hyx : y ≤ x)
    (hpal : ProjectEulerStatements.P4.isPalindrome (x * y)) :
    x * y ≤ (loopA a lo hi best bestA bestB).1 := by
  induction a, best, bestA, bestB using loopA.induct lo hi
    generalizing x y hlox hloy hyx hpal with
  | case1 a best bestA bestB ha =>
      omega
  | case2 a best bestA bestB ha hbest =>
      rw [loopA.eq_1]
      simp [ha, hbest]
      have hyhi : y ≤ hi := le_trans hyx (le_trans hxa hahi)
      have hprod : x * y ≤ a * hi := Nat.mul_le_mul hxa hyhi
      exact le_trans hprod (Nat.le_of_lt hbest)
  | case3 best bestA bestB best' bestA' bestB' ha hbest hloop =>
      have hx0 : x = 0 := by omega
      subst x
      simp
  | case4 a best bestA bestB ha hbest best' bestA' bestB' hloop hzero ih =>
      rw [loopA.eq_1]
      simp [ha, hbest, hloop, hzero]
      by_cases hxa' : x ≤ a - 1
      · exact ih x y (by omega) hlox hxa' hloy hyx hpal
      · have hx : x = a := by omega
        subst x
        have hB : a * y ≤ best' := by
          have hB0 := loopB_covers a a lo best bestA bestB y hloy hyx hpal
          rw [hloop] at hB0
          exact hB0
        exact le_trans hB (loopA_ge_best (a - 1) lo hi best' bestA' bestB')

lemma loopA_candidate (a lo hi best bestA bestB : Nat) :
    (loopA a lo hi best bestA bestB).1 = best ∨
      ∃ x y, lo ≤ x ∧ x ≤ a ∧ lo ≤ y ∧ y ≤ x ∧
        ProjectEulerStatements.P4.isPalindrome (x * y) ∧
        (loopA a lo hi best bestA bestB).1 = x * y := by
  induction a, best, bestA, bestB using loopA.induct lo hi with
  | case1 a best bestA bestB ha =>
      rw [loopA.eq_1]
      simp [ha]
  | case2 a best bestA bestB ha hbest =>
      rw [loopA.eq_1]
      simp [ha, hbest]
  | case3 best bestA bestB best' bestA' bestB' ha hbest hloop =>
      have hbest0 : best = 0 := by omega
      subst best
      left
      rw [loopA.eq_1]
      simp [ha, hloop]
      have hB0 : (loopB 0 0 lo 0 bestA bestB).1 = 0 := by
        rw [loopB.eq_1]
        simp [ha]
      rw [hloop] at hB0
      exact hB0
  | case4 a best bestA bestB ha hbest best' bestA' bestB' hloop hzero ih =>
      rw [loopA.eq_1]
      simp [ha, hbest, hloop, hzero]
      rcases ih with hA | hA
      · rcases loopB_candidate a a lo best bestA bestB with hB | hB
        · left
          rw [hloop] at hB
          exact hA.trans hB
        · rcases hB with ⟨y, hloy, hya, hpaly, hry⟩
          right
          refine Exists.intro a ?_
          refine ⟨by omega, by omega, ?_⟩
          refine Exists.intro y ?_
          refine ⟨hloy, hya, hpaly, ?_⟩
          rw [hloop] at hry
          exact hA.trans hry
      · rcases hA with ⟨x, y, hlox, hxa, hloy, hyx, hpaly, hry⟩
        right
        refine Exists.intro x ?_
        refine ⟨hlox, by omega, ?_⟩
        refine Exists.intro y ?_
        exact ⟨hloy, hyx, hpaly, hry⟩

lemma digitLower_le_digitUpper (n : Nat) :
    ProjectEulerStatements.P4.digitLower n ≤ ProjectEulerStatements.P4.digitUpper n := by
  cases n with
  | zero =>
      simp [ProjectEulerStatements.P4.digitLower, ProjectEulerStatements.P4.digitUpper]
  | succ d =>
      unfold ProjectEulerStatements.P4.digitLower ProjectEulerStatements.P4.digitUpper
      exact Nat.le_sub_one_of_lt (Nat.pow_lt_pow_succ (a := 10) (n := d) (by norm_num))

lemma solve_mem_or_zero (n : Nat) :
    solve n = 0 ∨ solve n ∈ ProjectEulerStatements.P4.palProductSet n := by
  unfold solve largestPalindromeProduct
  let lo := match n with | 0 => 0 | d + 1 => 10 ^ d
  let hi := 10 ^ n - 1
  have hlo_eq : lo = ProjectEulerStatements.P4.digitLower n := by
    cases n <;> rfl
  have hhi_eq : hi = ProjectEulerStatements.P4.digitUpper n := rfl
  rcases loopA_candidate hi lo hi 0 0 0 with hzero | hcand
  · left
    exact hzero
  · right
    rcases hcand with ⟨x, y, hlox, hxhi, hloy, hyx, hpal, hres⟩
    change (loopA hi lo hi 0 0 0).1 ∈ ProjectEulerStatements.P4.palProductSet n
    rw [hres]
    unfold ProjectEulerStatements.P4.palProductSet
    rw [Finset.mem_filter]
    constructor
    · rw [Finset.mem_image]
      refine ⟨(x, y), ?_, rfl⟩
      simp [Finset.mem_product, Finset.mem_Icc, ← hlo_eq, ← hhi_eq]
      constructor <;> omega
    · exact hpal

lemma naive_le_solve (n : Nat) : ProjectEulerStatements.P4.naive n ≤ solve n := by
  unfold ProjectEulerStatements.P4.naive
  by_cases h : (ProjectEulerStatements.P4.palProductSet n).Nonempty
  · rw [dif_pos h]
    rw [Finset.max'_le_iff]
    intro z hz
    unfold ProjectEulerStatements.P4.palProductSet at hz
    rw [Finset.mem_filter] at hz
    rcases hz with ⟨hz_image, hpal⟩
    rw [Finset.mem_image] at hz_image
    rcases hz_image with ⟨⟨x, y⟩, hxy, rfl⟩
    simp [Finset.mem_product, Finset.mem_Icc] at hxy
    rcases hxy with ⟨hx, hy⟩
    simp at hpal
    unfold solve largestPalindromeProduct
    let lo := match n with | 0 => 0 | d + 1 => 10 ^ d
    let hi := 10 ^ n - 1
    have hlo_eq : lo = ProjectEulerStatements.P4.digitLower n := by
      cases n <;> rfl
    have hhi_eq : hi = ProjectEulerStatements.P4.digitUpper n := rfl
    by_cases hyx : y ≤ x
    · exact loopA_covers hi lo hi 0 0 0 x y le_rfl (by omega) (by omega) (by omega) hyx hpal
    · have hxy_le : x ≤ y := by omega
      rw [Nat.mul_comm x y]
      exact loopA_covers hi lo hi 0 0 0 y x le_rfl (by omega) (by omega) (by omega) hxy_le
        (by simpa [Nat.mul_comm] using hpal)
  · rw [dif_neg h]
    exact Nat.zero_le _

lemma solve_le_naive (n : Nat) : solve n ≤ ProjectEulerStatements.P4.naive n := by
  rcases solve_mem_or_zero n with hzero | hmem
  · rw [hzero]
    exact Nat.zero_le _
  · unfold ProjectEulerStatements.P4.naive
    rw [dif_pos ⟨solve n, hmem⟩]
    exact Finset.le_max' (ProjectEulerStatements.P4.palProductSet n) (solve n) hmem

end Equiv

theorem equiv (n : Nat) : ProjectEulerStatements.P4.naive n = solve n := by
  exact le_antisymm (Equiv.naive_le_solve n) (Equiv.solve_le_naive n)

end ProjectEulerSolutions.P4
