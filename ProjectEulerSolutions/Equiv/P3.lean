import ProjectEulerSolutions.P3
import Mathlib.Data.Nat.Sqrt
import Mathlib.Tactic

namespace ProjectEulerSolutions.P3
open ProjectEulerStatements.P3

abbrev ResultOK (N r : Nat) : Prop :=
  isPrimeFactor N r ∧ ∀ p, isPrimeFactor N p → p ≤ r

abbrev LargestOK (N largest : Nat) : Prop :=
  largest = 1 ∨ isPrimeFactor N largest

abbrev Cover (N n largest : Nat) : Prop :=
  ∀ p, isPrimeFactor N p → p ≤ largest ∨ p ∣ n

abbrev Low (n f : Nat) : Prop :=
  ∀ p, Nat.Prime p → p ∣ n → f ≤ p

lemma prime_of_no_small_prime_divisor {n f : Nat} (hn : 1 < n)
    (hff : ¬ f * f ≤ n) (hlow : Low n f) : Nat.Prime n := by
  apply (Nat.prime_def_le_sqrt).2
  constructor
  · omega
  · intro m hm2 hmsqrt hmdvd
    obtain ⟨p, hp, hpdvd⟩ := Nat.exists_prime_and_dvd (by omega : m ≠ 1)
    have hpn : p ∣ n := dvd_trans hpdvd hmdvd
    have hpf : f ≤ p := hlow p hp hpn
    have hpm : p ≤ m := Nat.le_of_dvd (by omega : 0 < m) hpdvd
    have hmf : m < f := by
      by_contra hfm
      have hfsqrt : f ≤ Nat.sqrt n := le_trans (le_of_not_gt hfm) hmsqrt
      exact hff ((Nat.le_sqrt).1 hfsqrt)
    omega

lemma prime_of_divides_with_low {n f : Nat} (hf : 1 < f) (hfdvd : f ∣ n)
    (hlow : Low n f) : Nat.Prime f := by
  apply (Nat.prime_def_le_sqrt).2
  constructor
  · omega
  · intro m hm2 hmsqrt hmdvd
    obtain ⟨p, hp, hpdvd⟩ := Nat.exists_prime_and_dvd (by omega : m ≠ 1)
    have hpn : p ∣ n := dvd_trans (dvd_trans hpdvd hmdvd) hfdvd
    have hpf : f ≤ p := hlow p hp hpn
    have hpm : p ≤ m := Nat.le_of_dvd (by omega : 0 < m) hpdvd
    have hmf : m < f := lt_of_le_of_lt hmsqrt (Nat.sqrt_lt_self hf)
    omega

lemma lower_step {n f : Nat} (hfge : 3 ≤ f) (hfodd : Odd f)
    (hlow : Low n f) (hnotdvd : ¬ f ∣ n) : Low n (f + 2) := by
  intro p hp hpn
  have hfp : f ≤ p := hlow p hp hpn
  by_contra hlt
  have hp_eq : p = f ∨ p = f + 1 := by omega
  rcases hp_eq with rfl | hp_eq_succ
  · exact hnotdvd hpn
  · have hpne2 : p ≠ 2 := by omega
    have hpodd : Odd p := hp.odd_of_ne_two hpne2
    obtain ⟨a, ha⟩ := hfodd
    obtain ⟨b, hb⟩ := hpodd
    omega

lemma prime_dvd_div_of_ne {n f p : Nat} (hfdvd : f ∣ n) (hfp : Nat.Prime f)
    (hpp : Nat.Prime p) (hpn : p ∣ n) (hpf_gt : f < p) : p ∣ n / f := by
  obtain ⟨q, hq⟩ := hfdvd
  have hn_eq : n = f * q := by simpa [Nat.mul_comm] using hq
  have hdiv : n / f = q := by rw [hn_eq, Nat.mul_div_right _ hfp.pos]
  rw [hdiv]
  have hpmul : p ∣ f * q := by simpa [← hn_eq] using hpn
  rcases (Nat.Prime.dvd_mul hpp).1 hpmul with hpf_dvd | hpq
  · have hpeq : f = p := (Nat.Prime.dvd_iff_eq hfp hpp.ne_one).1 hpf_dvd
    omega
  · exact hpq

lemma result_of_state_final_large {N n f largest : Nat}
    (hnN : n ∣ N) (_hlargest : LargestOK N largest) (hcover : Cover N n largest)
    (hlow : Low n f) (hle_largest : n > 1 → largest ≤ n) (hn : n > 1)
    (hff : ¬ f * f ≤ n) : ResultOK N n := by
  have hnprime : Nat.Prime n := prime_of_no_small_prime_divisor hn hff hlow
  constructor
  · exact ⟨hnprime.two_le, hnN, hnprime⟩
  · intro p hpN
    rcases hcover p hpN with hple | hpdvdn
    · exact le_trans hple (hle_largest hn)
    · have hpeq : n = p := (Nat.Prime.dvd_iff_eq hnprime hpN.2.2.ne_one).1 hpdvdn
      omega

lemma result_of_state_final_small {N n largest : Nat}
    (_hN2 : 2 ≤ N) (hlargest : LargestOK N largest) (hcover : Cover N n largest)
    (hnpos : 0 < n) (hsmall : ¬ n > 1) (hnotone : n ≤ 1 → largest ≠ 1) :
    ResultOK N largest := by
  have hnle : n ≤ 1 := by omega
  have hne1 : largest ≠ 1 := hnotone hnle
  have hlpf : isPrimeFactor N largest := by
    rcases hlargest with h | h
    · exact (hne1 h).elim
    · exact h
  constructor
  · exact hlpf
  · intro p hpN
    rcases hcover p hpN with hple | hpdvdn
    · exact hple
    · have hpnle : p ≤ n := Nat.le_of_dvd hnpos hpdvdn
      have hpge : 2 ≤ p := hpN.1
      omega

lemma oddLoop_correct {N n f largest : Nat}
    (hN2 : 2 ≤ N) (hnN : n ∣ N) (hnpos : 0 < n)
    (hlargest : LargestOK N largest) (hcover : Cover N n largest)
    (hlow : Low n f) (hfge : 3 ≤ f) (hfodd : Odd f)
    (hlargest_le_f : largest ≤ f) (hle_largest : n > 1 → largest ≤ n)
    (hnotone : n ≤ 1 → largest ≠ 1) : ResultOK N (oddLoop n f largest) := by
  induction n, f, largest using oddLoop.induct with
  | case1 n f largest hf hn => omega
  | case2 n f largest hf hn => omega
  | case3 n f largest hfgt hff hmod ih =>
      rw [oddLoop.eq_def]
      simp [hfgt, hff, hmod]
      have hfdvd : f ∣ n := Nat.dvd_of_mod_eq_zero (by simpa using hmod)
      have hfprime : Nat.Prime f := prime_of_divides_with_low (by omega : 1 < f) hfdvd hlow
      have hndivN : n / f ∣ N := dvd_trans (Nat.div_dvd_of_dvd hfdvd) hnN
      have hn_div_pos : 0 < n / f := Nat.div_pos (by
        exact le_trans (Nat.le_mul_of_pos_right f (by omega : 0 < f)) hff) (by omega : 0 < f)
      have hlargest' : LargestOK N f := Or.inr ⟨hfprime.two_le, dvd_trans hfdvd hnN, hfprime⟩
      have hcover' : Cover N (n / f) f := by
        intro p hpN
        rcases hcover p hpN with hple | hpdvdn
        · exact Or.inl (le_trans hple hlargest_le_f)
        · by_cases hpfle : p ≤ f
          · exact Or.inl hpfle
          · exact Or.inr (prime_dvd_div_of_ne hfdvd hfprime hpN.2.2 hpdvdn (by omega))
      have hlow' : Low (n / f) f := by
        intro p hp hpdvd
        exact hlow p hp (dvd_trans hpdvd (Nat.div_dvd_of_dvd hfdvd))
      have hle_largest' : n / f > 1 → f ≤ n / f := by
        intro hgt
        obtain ⟨q, hq⟩ := hfdvd
        have hn_eq : n = f * q := by simpa [Nat.mul_comm] using hq
        have hqeq : n / f = q := by rw [hn_eq, Nat.mul_div_right _ hfprime.pos]
        rw [hqeq]
        nlinarith [hff, show n = f * q by exact hn_eq]
      have hnotone' : n / f ≤ 1 → f ≠ 1 := by intro _; omega
      exact ih hndivN hn_div_pos hlargest' hcover' hlow' hfge hfodd (le_rfl)
        hle_largest' hnotone'
  | case4 n f largest hfgt hff hmod ih =>
      rw [oddLoop.eq_def]
      simp [hfgt, hff, hmod]
      have hnotdvd : ¬ f ∣ n := by
        intro hd
        have hm : n % f = 0 := Nat.mod_eq_zero_of_dvd hd
        exact hmod (by simp [hm])
      have hlow' : Low n (f + 2) := lower_step hfge hfodd hlow hnotdvd
      have hfodd' : Odd (f + 2) := by
        obtain ⟨a, ha⟩ := hfodd
        use a + 1
        omega
      exact ih hnN hnpos hlargest hcover hlow' (by omega) hfodd' (by omega)
        hle_largest hnotone
  | case5 n f largest hfgt hff hn =>
      rw [oddLoop.eq_def]
      simp [hfgt, hff, hn]
      exact result_of_state_final_large hnN hlargest hcover hlow hle_largest hn hff
  | case6 n f largest hfgt hff hn =>
      rw [oddLoop.eq_def]
      simp [hfgt, hff, hn]
      exact result_of_state_final_small hN2 hlargest hcover hnpos hn hnotone

abbrev StripOK (N : Nat) (r : Nat × Nat) : Prop :=
  r.1 ∣ N ∧ 0 < r.1 ∧ LargestOK N r.2 ∧ Cover N r.1 r.2 ∧ Low r.1 3 ∧
    r.2 ≤ 3 ∧ (r.1 > 1 → r.2 ≤ r.1) ∧ (r.1 ≤ 1 → r.2 ≠ 1)

lemma stripTwos_correct {N n largest : Nat}
    (hnN : n ∣ N) (hnpos : 0 < n) (hlargest : LargestOK N largest)
    (hcover : Cover N n largest) (hlargest_le_two : largest ≤ 2)
    (hle_largest : n > 1 → largest ≤ n) (hnotone : n ≤ 1 → largest ≠ 1) :
    StripOK N (stripTwos n largest) := by
  induction n, largest using stripTwos.induct with
  | case1 largest => omega
  | case2 n largest hn0 hmod ih =>
      rw [stripTwos.eq_def]
      simp [hn0, hmod]
      have h2dvd : 2 ∣ n := Nat.dvd_of_mod_eq_zero (by simpa using hmod)
      have hn_ne_one : n ≠ 1 := by
        intro h
        subst n
        simp at hmod
      have hn2 : 2 ≤ n := by omega
      have hndivN : n / 2 ∣ N := dvd_trans (Nat.div_dvd_of_dvd h2dvd) hnN
      have hndivpos : 0 < n / 2 := Nat.div_pos hn2 (by norm_num)
      have hlargest' : LargestOK N 2 := by
        exact Or.inr ⟨by norm_num, dvd_trans h2dvd hnN, Nat.prime_two⟩
      have hcover' : Cover N (n / 2) 2 := by
        intro p hpN
        rcases hcover p hpN with hple | hpdvdn
        · exact Or.inl (le_trans hple hlargest_le_two)
        · by_cases hp2 : p = 2
          · exact Or.inl (by omega)
          · have hpdiv : p ∣ n / 2 := by
              obtain ⟨q, hq⟩ := h2dvd
              have hn_eq : n = 2 * q := by simpa [Nat.mul_comm] using hq
              have hdiv_eq : n / 2 = q := by
                rw [hn_eq, Nat.mul_div_right _ (by norm_num : 0 < 2)]
              rw [hdiv_eq]
              have hpmul : p ∣ 2 * q := by simpa [← hn_eq] using hpdvdn
              rcases (Nat.Prime.dvd_mul hpN.2.2).1 hpmul with hp_dvd_two | hpq
              · have hpeq : 2 = p :=
                  (Nat.Prime.dvd_iff_eq Nat.prime_two hpN.2.2.ne_one).1 hp_dvd_two
                omega
              · exact hpq
            exact Or.inr hpdiv
      have hle_largest' : n / 2 > 1 → 2 ≤ n / 2 := by omega
      have hnotone' : n / 2 ≤ 1 → 2 ≠ 1 := by omega
      exact ih hndivN hndivpos hlargest' hcover' (by norm_num) hle_largest' hnotone'
  | case3 n largest hn0 hnotmod =>
      rw [stripTwos.eq_def]
      simp [hn0, hnotmod]
      have hlow : Low n 3 := by
        intro p hp hpdvd
        by_cases hp2 : p = 2
        · subst p
          have hm : n % 2 = 0 := Nat.mod_eq_zero_of_dvd hpdvd
          exact (hnotmod (by simp [hm])).elim
        · have hp2le : 2 ≤ p := hp.two_le
          omega
      exact ⟨hnN, hnpos, hlargest, hcover, hlow, le_trans hlargest_le_two (by norm_num),
        hle_largest, hnotone⟩

lemma resultOK_findGreatest {N r : Nat} (hNpos : 0 < N) (hok : ResultOK N r) :
    Nat.findGreatest (isPrimeFactor N) N = r := by
  apply (Nat.findGreatest_eq_iff (P := isPrimeFactor N) (k := N) (m := r)).2
  have hrle : r ≤ N := Nat.le_of_dvd hNpos hok.1.2.1
  refine ⟨hrle, ?_, ?_⟩
  · intro _
    exact hok.1
  · intro k hk hkN hPk
    exact not_le_of_gt hk (hok.2 k hPk)

lemma solve_resultOK (n : ProjectEulerStatements.P3.NatGE2) : ResultOK n.1 (solve n) := by
  unfold solve
  generalize hst : stripTwos n.1 1 = st
  rcases st with ⟨n', largest'⟩
  have hstrip : StripOK n.1 (n', largest') := by
    have hnpos : 0 < n.1 := lt_of_lt_of_le (by norm_num : 0 < 2) n.2
    have hcover : Cover n.1 n.1 1 := by
      intro p hp
      exact Or.inr hp.2.1
    have hstrip0 : StripOK n.1 (stripTwos n.1 1) :=
      stripTwos_correct (N := n.1) (n := n.1) (largest := 1) dvd_rfl hnpos
        (Or.inl rfl) hcover (by norm_num) (by intro hn; omega)
        (by
          intro hnle _
          have hnnot : ¬ n.1 ≤ 1 := not_le_of_gt (lt_of_lt_of_le (by norm_num : 1 < 2) n.2)
          exact (hnnot hnle).elim)
    simpa [hst] using hstrip0
  exact oddLoop_correct (N := n.1) (n := n') (f := 3) (largest := largest') n.2
    hstrip.1 hstrip.2.1 hstrip.2.2.1 hstrip.2.2.2.1 hstrip.2.2.2.2.1
    (by norm_num) (by norm_num : Odd 3) hstrip.2.2.2.2.2.1 hstrip.2.2.2.2.2.2.1
    hstrip.2.2.2.2.2.2.2

theorem equiv (n : ProjectEulerStatements.P3.NatGE2) :
    ProjectEulerStatements.P3.naive n = solve n := by
  unfold ProjectEulerStatements.P3.naive
  exact resultOK_findGreatest (lt_of_lt_of_le (by norm_num : 0 < 2) n.2) (solve_resultOK n)

end ProjectEulerSolutions.P3
