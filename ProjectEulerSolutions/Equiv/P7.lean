import ProjectEulerSolutions.P7
import Mathlib.NumberTheory.Bertrand
import Mathlib.NumberTheory.PrimeCounting
import Mathlib.Tactic

namespace ProjectEulerSolutions.P7

lemma nth_prime_le_two_pow_succ (n : Nat) : Nat.nth Nat.Prime n ≤ 2 ^ (n + 1) := by
  induction n with
  | zero => simp
  | succ n ih =>
      have hpos : Nat.nth Nat.Prime n ≠ 0 := (Nat.prime_nth_prime n).ne_zero
      rcases Nat.exists_prime_lt_and_le_two_mul (Nat.nth Nat.Prime n) hpos with
        ⟨p, hp, _hgt, hle⟩
      have hp_range : p ∈ Set.range (Nat.nth Nat.Prime) := by
        rw [Nat.range_nth_of_infinite Nat.infinite_setOf_prime]
        exact hp
      rcases hp_range with ⟨k, hk⟩
      have hnlt : n < k := by
        by_contra hnk
        have hk_le_n : k ≤ n := Nat.le_of_not_gt hnk
        have hle_nth : Nat.nth Nat.Prime k ≤ Nat.nth Nat.Prime n :=
          (Nat.nth_le_nth Nat.infinite_setOf_prime).2 hk_le_n
        omega
      have hn1le : n + 1 ≤ k := Nat.succ_le_of_lt hnlt
      have hnext_le_p : Nat.nth Nat.Prime (n + 1) ≤ p := by
        rw [← hk]
        exact (Nat.nth_le_nth Nat.infinite_setOf_prime).2 hn1le
      calc
        Nat.nth Nat.Prime (n + 1) ≤ p := hnext_le_p
        _ ≤ 2 * Nat.nth Nat.Prime n := hle
        _ ≤ 2 * 2 ^ (n + 1) := Nat.mul_le_mul_left 2 ih
        _ = 2 ^ (n + 2) := by ring

lemma one_le_initialLimit (n : Nat) : 1 ≤ initialLimit n := by
  unfold initialLimit
  split <;> omega

lemma initialSieve_size (limit : Nat) :
    (initialSieve limit).size = sieveSize limit := by
  simp [initialSieve]

lemma markMultiples_size (p : Nat) (arr : Array Bool) :
    (markMultiples p arr).size = arr.size := by
  simp [markMultiples]

lemma sieveState_size (limit processed : Nat) :
    (sieveState limit processed).size = sieveSize limit := by
  simp [sieveState]

lemma sieveState_get! (limit processed idx : Nat)
    (hidx : idx < (sieveState limit processed).size) :
    (sieveState limit processed)[idx]! =
      decide (idx != 0 ∧ ¬ hasMarkedFactorBefore processed idx) := by
  calc
    (sieveState limit processed)[idx]! = (sieveState limit processed)[idx] :=
      getElem!_pos (sieveState limit processed) idx hidx
    _ = decide (idx != 0 ∧ ¬ hasMarkedFactorBefore processed idx) := by
      simp [sieveState]

lemma markMultiples_get! (p idx : Nat) (arr : Array Bool)
    (hidx : idx < (markMultiples p arr).size) :
    (markMultiples p arr)[idx]! =
      if p * p ≤ oddAt idx ∧ oddAt idx % p = 0 then false else arr[idx]! := by
  have hidx_arr : idx < arr.size := by
    simpa [markMultiples_size] using hidx
  calc
    (markMultiples p arr)[idx]! = (markMultiples p arr)[idx] :=
      getElem!_pos (markMultiples p arr) idx hidx
    _ = (if p * p ≤ oddAt idx ∧ oddAt idx % p = 0 then false else arr[idx]!) := by
      rw [show arr[idx]! = arr[idx] by exact getElem!_pos arr idx hidx_arr]
      simp [markMultiples]

lemma initialSieve_eq_sieveState_one (limit : Nat) :
    initialSieve limit = sieveState limit 1 := by
  apply Array.ext
  · rw [initialSieve_size, sieveState_size]
  · intro idx hinit hstate
    simp [initialSieve, sieveState, hasMarkedFactorBefore, markedByIndex]

lemma hasMarkedFactorBefore_succ (i idx : Nat) :
    hasMarkedFactorBefore (i + 1) idx ↔
      hasMarkedFactorBefore i idx ∨ markedByIndex i idx := by
  constructor
  · intro h
    rcases h with ⟨j, hj, hmark⟩
    have hjlt : j < i + 1 := by simpa using hj
    by_cases hji : j < i
    · left
      exact ⟨j, by simpa using hji, hmark⟩
    · right
      have hjeq : j = i := by omega
      simpa [hjeq] using hmark
  · intro h
    rcases h with h | h
    · rcases h with ⟨j, hj, hmark⟩
      exact ⟨j, by simp_all [Finset.mem_range]; omega, hmark⟩
    · exact ⟨i, by simp, h⟩

lemma mark_condition_iff_marked {i idx : Nat} (hi : 1 ≤ i) :
    (oddAt i * oddAt i ≤ oddAt idx ∧ oddAt idx % oddAt i = 0) ↔
      markedByIndex i idx := by
  constructor
  · intro h
    exact ⟨hi, h.1, Nat.dvd_of_mod_eq_zero h.2⟩
  · intro h
    exact ⟨h.2.1, Nat.mod_eq_zero_of_dvd h.2.2⟩

lemma markMultiples_sieveState_live (limit i : Nat)
    (hi_size : i < (sieveState limit i).size)
    (hlive : (sieveState limit i)[i]! = true) :
    markMultiples (oddAt i) (sieveState limit i) = sieveState limit (i + 1) := by
  have hlive_prop : i != 0 ∧ ¬ hasMarkedFactorBefore i i := by
    have hget := sieveState_get! limit i i hi_size
    exact of_decide_eq_true (by simpa [hget] using hlive)
  have hi_pos : 1 ≤ i := by
    have hne : i ≠ 0 := by
      intro h
      simp [h] at hlive_prop
    omega
  apply Array.ext
  · rw [markMultiples_size, sieveState_size, sieveState_size]
  · intro idx hmark hstate
    have hidx_state_i : idx < (sieveState limit i).size := by
      simpa [markMultiples_size] using hmark
    rw [← getElem!_pos (markMultiples (oddAt i) (sieveState limit i)) idx hmark]
    rw [← getElem!_pos (sieveState limit (i + 1)) idx hstate]
    rw [markMultiples_get! (oddAt i) idx (sieveState limit i) hmark]
    rw [sieveState_get! limit (i + 1) idx hstate]
    rw [sieveState_get! limit i idx hidx_state_i]
    by_cases hcond : oddAt i * oddAt i ≤ oddAt idx ∧ oddAt idx % oddAt i = 0
    · have hmarked : markedByIndex i idx := (mark_condition_iff_marked hi_pos).1 hcond
      have hbefore_succ : hasMarkedFactorBefore (i + 1) idx := by
        exact (hasMarkedFactorBefore_succ i idx).2 (Or.inr hmarked)
      simp [hcond, hbefore_succ]
    · have hnot_marked : ¬ markedByIndex i idx := by
        intro hm
        exact hcond ((mark_condition_iff_marked hi_pos).2 hm)
      have hnot_before_succ :
          ¬ hasMarkedFactorBefore (i + 1) idx ↔ ¬ hasMarkedFactorBefore i idx := by
        rw [hasMarkedFactorBefore_succ]
        constructor
        · intro h hb
          exact h (Or.inl hb)
        · intro h hb
          rcases hb with hb | hb
          · exact h hb
          · exact hnot_marked hb
      by_cases hidx0 : idx = 0
      · subst idx
        simp [hcond]
      · have hidx_ne : (idx != 0) = true := by simp [hidx0]
        by_cases hbefore : hasMarkedFactorBefore i idx
        · have hbefore_succ : hasMarkedFactorBefore (i + 1) idx :=
            (hasMarkedFactorBefore_succ i idx).2 (Or.inl hbefore)
          simp [hcond, hidx_ne, hbefore, hbefore_succ]
        · have hnot_succ : ¬ hasMarkedFactorBefore (i + 1) idx :=
            hnot_before_succ.2 hbefore
          simp [hcond, hidx_ne, hbefore, hnot_succ]

lemma hasMarkedFactorBefore_of_dead_marker {i idx : Nat}
    (hdead : hasMarkedFactorBefore i i) (hmark : markedByIndex i idx) :
    hasMarkedFactorBefore i idx := by
  rcases hdead with ⟨j, hj, hjmark⟩
  rcases hjmark with ⟨hjpos, hjsq_le_i, hjdvd_i⟩
  rcases hmark with ⟨hipos, hisq_le_idx, hidvd_idx⟩
  refine ⟨j, hj, ?_⟩
  refine ⟨hjpos, ?_, dvd_trans hjdvd_i hidvd_idx⟩
  have hi_le_sq : oddAt i ≤ oddAt i * oddAt i := by
    have : 1 ≤ oddAt i := by unfold oddAt; omega
    nlinarith
  exact le_trans hjsq_le_i (le_trans hi_le_sq hisq_le_idx)

lemma exists_prime_dvd_sq_le_of_not_prime {n : Nat} (hn2 : 2 ≤ n) (hnp : ¬ Nat.Prime n) :
    ∃ p, Nat.Prime p ∧ p ∣ n ∧ p * p ≤ n := by
  have hnotall : ¬ ∀ m, 2 ≤ m → m ≤ Nat.sqrt n → ¬ m ∣ n := by
    intro h
    exact hnp ((Nat.prime_def_le_sqrt).2 ⟨hn2, h⟩)
  push_neg at hnotall
  rcases hnotall with ⟨m, hm2, hmsqrt, hmdvd⟩
  obtain ⟨p, hp, hpdvdm⟩ := Nat.exists_prime_and_dvd (by omega : m ≠ 1)
  refine ⟨p, hp, dvd_trans hpdvdm hmdvd, ?_⟩
  have hpm : p ≤ m := Nat.le_of_dvd (by omega : 0 < m) hpdvdm
  have hpsqrt : p ≤ Nat.sqrt n := le_trans hpm hmsqrt
  exact (Nat.le_sqrt).1 hpsqrt

lemma not_hasMarkedFactorBefore_of_prime {processed idx : Nat}
    (hp : Nat.Prime (oddAt idx)) :
    ¬ hasMarkedFactorBefore processed idx := by
  intro h
  rcases h with ⟨j, _hj, hjpos, hjsq, hjdvd⟩
  have hdiv_eq : oddAt j = oddAt idx := by
    have hcases := hp.eq_one_or_self_of_dvd (oddAt j) hjdvd
    rcases hcases with hone | hself
    · have hjge : 3 ≤ oddAt j := by unfold oddAt; omega
      omega
    · exact hself
  have hidx_ge : 2 ≤ oddAt idx := hp.two_le
  have hgt : oddAt idx < oddAt idx * oddAt idx := by
    nlinarith
  rw [hdiv_eq] at hjsq
  omega

lemma hasMarkedFactorBefore_of_not_prime_of_le_limit {limit idx : Nat}
    (hidx_pos : 1 ≤ idx)
    (hidx_le : oddAt idx ≤ limit)
    (hnp : ¬ Nat.Prime (oddAt idx)) :
    hasMarkedFactorBefore (limit.sqrt / 2 + 1) idx := by
  have hn2 : 2 ≤ oddAt idx := by unfold oddAt; omega
  rcases exists_prime_dvd_sq_le_of_not_prime hn2 hnp with ⟨p, hp, hpdvd, hpsq⟩
  have hp_ne_two : p ≠ 2 := by
    intro hp2
    subst p
    rcases hpdvd with ⟨k, hk⟩
    unfold oddAt at hk
    omega
  rcases hp.odd_of_ne_two hp_ne_two with ⟨j, hj⟩
  have hp_eq : p = oddAt j := by
    unfold oddAt
    exact hj
  have hj_pos : 1 ≤ j := by
    have hp2le := hp.two_le
    rw [hp_eq] at hp2le
    unfold oddAt at hp2le
    omega
  have hp_le_sqrt : p ≤ limit.sqrt := by
    apply (Nat.le_sqrt).2
    exact le_trans hpsq hidx_le
  have hj_range : j < limit.sqrt / 2 + 1 := by
    rw [hp_eq] at hp_le_sqrt
    unfold oddAt at hp_le_sqrt
    omega
  refine ⟨j, by simpa using hj_range, ?_⟩
  refine ⟨hj_pos, ?_, ?_⟩
  · simpa [hp_eq] using hpsq
  · simpa [hp_eq] using hpdvd

lemma odd_prime_index_lt_processed_of_sq_le_succ {limit p j : Nat}
    (hp_eq : p = oddAt j) (hpsq : p * p ≤ limit + 1) :
    j < limit.sqrt / 2 + 1 := by
  by_contra hnot
  have hjge : limit.sqrt / 2 + 1 ≤ j := Nat.le_of_not_gt hnot
  have hrootp : limit.sqrt + 1 < p := by
    rw [hp_eq]
    unfold oddAt
    omega
  have hroot_lt : limit.sqrt < p - 1 := by omega
  have hlimit_lt : limit < (p - 1) * (p - 1) := by
    exact (Nat.sqrt_lt).1 hroot_lt
  have hlimit_succ_le : limit + 1 ≤ (p - 1) * (p - 1) := by omega
  have hp_pos : 0 < p := by
    rw [hp_eq]
    unfold oddAt
    omega
  have hp_pred : p - 1 + 1 = p := Nat.succ_pred_eq_of_pos hp_pos
  have hsq_lt : (p - 1) * (p - 1) < p * p := by
    nlinarith
  omega

lemma hasMarkedFactorBefore_of_not_prime_of_idx_lt_size {limit idx : Nat}
    (hidx_pos : 1 ≤ idx)
    (hidx_size : idx < sieveSize limit)
    (hnp : ¬ Nat.Prime (oddAt idx)) :
    hasMarkedFactorBefore (limit.sqrt / 2 + 1) idx := by
  have hn2 : 2 ≤ oddAt idx := by unfold oddAt; omega
  rcases exists_prime_dvd_sq_le_of_not_prime hn2 hnp with ⟨p, hp, hpdvd, hpsq⟩
  have hidx_le_succ : oddAt idx ≤ limit + 1 := by
    unfold sieveSize at hidx_size
    unfold oddAt
    omega
  have hp_ne_two : p ≠ 2 := by
    intro hp2
    subst p
    rcases hpdvd with ⟨k, hk⟩
    unfold oddAt at hk
    omega
  rcases hp.odd_of_ne_two hp_ne_two with ⟨j, hj⟩
  have hp_eq : p = oddAt j := by
    unfold oddAt
    exact hj
  have hj_pos : 1 ≤ j := by
    have hp2le := hp.two_le
    rw [hp_eq] at hp2le
    unfold oddAt at hp2le
    omega
  have hj_range : j < limit.sqrt / 2 + 1 :=
    odd_prime_index_lt_processed_of_sq_le_succ hp_eq (le_trans hpsq hidx_le_succ)
  refine ⟨j, by simpa using hj_range, ?_⟩
  refine ⟨hj_pos, ?_, ?_⟩
  · simpa [hp_eq] using hpsq
  · simpa [hp_eq] using hpdvd

lemma sieveState_final_get!_of_le_limit {limit idx : Nat}
    (hidx_size : idx < (sieveState limit (limit.sqrt / 2 + 1)).size)
    (hidx_le : oddAt idx ≤ limit) :
    (sieveState limit (limit.sqrt / 2 + 1))[idx]! =
      decide (idx != 0 ∧ Nat.Prime (oddAt idx)) := by
  rw [sieveState_get! limit (limit.sqrt / 2 + 1) idx hidx_size]
  by_cases hidx0 : idx = 0
  · subst idx
    simp [oddAt]
  · have hidx_ne : (idx != 0) = true := by simp [hidx0]
    have hidx_pos : 1 ≤ idx := by omega
    by_cases hp : Nat.Prime (oddAt idx)
    · have hnot_marked := not_hasMarkedFactorBefore_of_prime
        (processed := limit.sqrt / 2 + 1) (idx := idx) hp
      simp [hidx_ne, hp, hnot_marked]
    · have hmarked := hasMarkedFactorBefore_of_not_prime_of_le_limit
        (limit := limit) (idx := idx) hidx_pos hidx_le hp
      simp [hidx_ne, hp, hmarked]

lemma sieveState_final_get! {limit idx : Nat}
    (hidx_size : idx < (sieveState limit (limit.sqrt / 2 + 1)).size) :
    (sieveState limit (limit.sqrt / 2 + 1))[idx]! =
      decide (idx != 0 ∧ Nat.Prime (oddAt idx)) := by
  rw [sieveState_get! limit (limit.sqrt / 2 + 1) idx hidx_size]
  by_cases hidx0 : idx = 0
  · subst idx
    simp [oddAt]
  · have hidx_ne : (idx != 0) = true := by simp [hidx0]
    have hidx_pos : 1 ≤ idx := by omega
    have hsize : idx < sieveSize limit := by
      simpa [sieveState_size] using hidx_size
    by_cases hp : Nat.Prime (oddAt idx)
    · have hnot_marked := not_hasMarkedFactorBefore_of_prime
        (processed := limit.sqrt / 2 + 1) (idx := idx) hp
      simp [hidx_ne, hp, hnot_marked]
    · have hmarked := hasMarkedFactorBefore_of_not_prime_of_idx_lt_size
        (limit := limit) (idx := idx) hidx_pos hsize hp
      simp [hidx_ne, hp, hmarked]

lemma sieveState_dead_eq_succ (limit i : Nat)
    (hi_size : i < (sieveState limit i).size)
    (hi_pos : 1 ≤ i)
    (hdeadBool : ¬ (sieveState limit i)[i]! = true) :
    sieveState limit i = sieveState limit (i + 1) := by
  have hdead : hasMarkedFactorBefore i i := by
    have hget := sieveState_get! limit i i hi_size
    have hnot : ¬ ((i != 0) = true ∧ ¬ hasMarkedFactorBefore i i) := by
      intro h
      have htrue : (sieveState limit i)[i]! = true := by
        rw [hget]
        exact decide_eq_true h
      exact hdeadBool htrue
    have hi_ne : (i != 0) = true := by
      simp [show i ≠ 0 by omega]
    by_contra hbefore
    exact hnot ⟨hi_ne, hbefore⟩
  apply Array.ext
  · rw [sieveState_size, sieveState_size]
  · intro idx hleft hright
    rw [← getElem!_pos (sieveState limit i) idx hleft]
    rw [← getElem!_pos (sieveState limit (i + 1)) idx hright]
    rw [sieveState_get! limit i idx hleft]
    rw [sieveState_get! limit (i + 1) idx hright]
    have hiff : hasMarkedFactorBefore (i + 1) idx ↔ hasMarkedFactorBefore i idx := by
      rw [hasMarkedFactorBefore_succ]
      constructor
      · intro h
        rcases h with h | h
        · exact h
        · exact hasMarkedFactorBefore_of_dead_marker hdead h
      · intro h
        exact Or.inl h
    by_cases hidx0 : idx = 0
    · subst idx
      simp
    · have hidx_ne : (idx != 0) = true := by simp [hidx0]
      by_cases hbefore : hasMarkedFactorBefore i idx
      · have hbefore_succ : hasMarkedFactorBefore (i + 1) idx := hiff.2 hbefore
        simp [hidx_ne, hbefore, hbefore_succ]
      · have hnot_succ : ¬ hasMarkedFactorBefore (i + 1) idx := by
          intro h
          exact hbefore (hiff.1 h)
        simp [hidx_ne, hbefore, hnot_succ]

lemma sieveLoop_sieveState_aux (limit maxI i : Nat) (isPrime : Array Bool)
    (hmax_size : maxI < sieveSize limit)
    (hstate : isPrime = sieveState limit i)
    (hi_pos : 1 ≤ i)
    (hi_le : i ≤ maxI + 1) :
    sieveLoop limit maxI i isPrime = sieveState limit (maxI + 1) := by
  induction i, isPrime using sieveLoop.induct maxI with
  | case1 i isPrime hgt =>
      have hi_eq : i = maxI + 1 := by omega
      rw [sieveLoop.eq_def]
      simp [hgt]
      rw [hstate, hi_eq]
  | case2 i isPrime hnot hprime p ih =>
      subst isPrime
      have hi_le_max : i ≤ maxI := by omega
      have hi_size : i < (sieveState limit i).size := by
        rw [sieveState_size]
        omega
      rw [sieveLoop.eq_def]
      simp [hnot, hprime]
      have hmark := markMultiples_sieveState_live limit i hi_size hprime
      exact ih hmark (by omega) (by omega)
  | case3 i isPrime hnot hprime ih =>
      subst isPrime
      have hi_le_max : i ≤ maxI := by omega
      have hi_size : i < (sieveState limit i).size := by
        rw [sieveState_size]
        omega
      rw [sieveLoop.eq_def]
      simp [hnot, hprime]
      have hskip := sieveState_dead_eq_succ limit i hi_size hi_pos hprime
      exact ih hskip (by omega) (by omega)

lemma sieveLoop_sieveState (limit maxI : Nat)
    (hmax_size : maxI < sieveSize limit) :
    sieveLoop limit maxI 1 (sieveState limit 1) = sieveState limit (maxI + 1) := by
  exact sieveLoop_sieveState_aux limit maxI 1 (sieveState limit 1)
    hmax_size rfl (by omega) (by omega)

lemma maxI_lt_sieveSize (limit : Nat) : limit.sqrt / 2 < sieveSize limit := by
  have hsqrt : limit.sqrt ≤ limit := Nat.sqrt_le_self limit
  unfold sieveSize
  omega

lemma oddsOnlySieveImpl_eq_state (limit : Nat) :
    oddsOnlySieveImpl limit = sieveState limit (limit.sqrt / 2 + 1) := by
  unfold oddsOnlySieveImpl
  rw [initialSieve_eq_sieveState_one]
  exact sieveLoop_sieveState limit (limit.sqrt / 2) (maxI_lt_sieveSize limit)

lemma oddsOnlySieveImpl_size (limit : Nat) :
    (oddsOnlySieveImpl limit).size = sieveSize limit := by
  rw [oddsOnlySieveImpl_eq_state, sieveState_size]

lemma oddsOnlySieveImpl_get!_of_le_limit {limit idx : Nat}
    (hidx_size : idx < (oddsOnlySieveImpl limit).size)
    (hidx_le : oddAt idx ≤ limit) :
    (oddsOnlySieveImpl limit)[idx]! =
      decide (idx != 0 ∧ Nat.Prime (oddAt idx)) := by
  rw [oddsOnlySieveImpl_eq_state] at hidx_size ⊢
  exact sieveState_final_get!_of_le_limit hidx_size hidx_le

lemma oddsOnlySieveImpl_get! {limit idx : Nat}
    (hidx_size : idx < (oddsOnlySieveImpl limit).size) :
    (oddsOnlySieveImpl limit)[idx]! =
      decide (idx != 0 ∧ Nat.Prime (oddAt idx)) := by
  rw [oddsOnlySieveImpl_eq_state] at hidx_size ⊢
  exact sieveState_final_get! hidx_size

lemma sieveLoop_size (limit maxI i : Nat) (isPrime : Array Bool) :
    (sieveLoop limit maxI i isPrime).size = isPrime.size := by
  induction i, isPrime using sieveLoop.induct maxI with
  | case1 i isPrime hgt =>
      rw [sieveLoop.eq_def]
      simp [hgt]
  | case2 i isPrime hnot hprime p ih =>
      rw [sieveLoop.eq_def]
      simp [hnot, hprime]
      rw [ih, markMultiples_size]
  | case3 i isPrime hnot hprime ih =>
      rw [sieveLoop.eq_def]
      simp [hnot, hprime]
      exact ih

lemma oddsOnlySieve_size (limit : Nat) :
    (oddsOnlySieve limit).size = sieveSize limit := by
  unfold oddsOnlySieve
  exact oddsOnlySieveImpl_size limit

lemma oddsOnlySieve_get! (limit i : Nat) (hi : i < (oddsOnlySieve limit).size) :
    (oddsOnlySieve limit)[i]! = decide (i != 0 ∧ Nat.Prime (oddAt i)) := by
  unfold oddsOnlySieve at hi ⊢
  exact oddsOnlySieveImpl_get! hi

lemma oddAt_succ (i : Nat) : oddAt (i + 1) = oddAt i + 2 := by
  unfold oddAt
  omega

lemma not_prime_even_gt_two {n : Nat} (h2 : 2 < n) (heven : Even n) : ¬ Nat.Prime n := by
  intro hp
  have hodd := hp.odd_of_ne_two (by omega)
  exact Nat.not_even_iff_odd.mpr hodd heven

lemma not_prime_oddAt_add_one {i : Nat} (hi : 1 ≤ i) : ¬ Nat.Prime (oddAt i + 1) := by
  apply not_prime_even_gt_two
  · unfold oddAt
    omega
  · unfold oddAt
    use i + 1
    omega

lemma count_prime_oddAt_succ_of_prime {i : Nat} (hi : 1 ≤ i)
    (hp : Nat.Prime (oddAt i)) :
    Nat.count Nat.Prime (oddAt (i + 1)) = Nat.count Nat.Prime (oddAt i) + 1 := by
  rw [oddAt_succ]
  rw [show oddAt i + 2 = (oddAt i + 1) + 1 by omega]
  rw [(Nat.count_succ_eq_count_iff (p := Nat.Prime) (n := oddAt i + 1)).2
    (not_prime_oddAt_add_one hi)]
  exact (Nat.count_succ_eq_succ_count_iff (p := Nat.Prime) (n := oddAt i)).2 hp

lemma count_prime_oddAt_succ_of_not_prime {i : Nat} (hi : 1 ≤ i)
    (hnp : ¬ Nat.Prime (oddAt i)) :
    Nat.count Nat.Prime (oddAt (i + 1)) = Nat.count Nat.Prime (oddAt i) := by
  rw [oddAt_succ]
  rw [show oddAt i + 2 = (oddAt i + 1) + 1 by omega]
  rw [(Nat.count_succ_eq_count_iff (p := Nat.Prime) (n := oddAt i + 1)).2
    (not_prime_oddAt_add_one hi)]
  exact (Nat.count_succ_eq_count_iff (p := Nat.Prime) (n := oddAt i)).2 hnp

lemma sieve_index_lt_of_oddAt_le {limit i : Nat} (h : oddAt i ≤ limit) :
    i < (oddsOnlySieve limit).size := by
  rw [oddsOnlySieve_size]
  unfold sieveSize oddAt at *
  omega

lemma oddAt_pos_index_prime_iff {limit i : Nat} (hi : i < (oddsOnlySieve limit).size)
    (hpos : 1 ≤ i) :
    (oddsOnlySieve limit)[i]! = true ↔ Nat.Prime (oddAt i) := by
  rw [oddsOnlySieve_get! limit i hi]
  constructor
  · intro h
    have hp : i != 0 ∧ Nat.Prime (oddAt i) := of_decide_eq_true h
    exact hp.2
  · intro hp
    apply decide_eq_true
    exact ⟨by simp [show i ≠ 0 by omega], hp⟩

lemma scanNthPrimeIndex_eq_some (limit k i count : Nat)
    (hbound : Nat.nth Nat.Prime k ≤ limit)
    (hcount : count = Nat.count Nat.Prime (oddAt i))
    (hi_pos : 1 ≤ i)
    (hcandidate : oddAt i ≤ Nat.nth Nat.Prime k) :
    scanNthPrimeIndex k i count (oddsOnlySieve limit) = some (Nat.nth Nat.Prime k) := by
  induction i, count using scanNthPrimeIndex.induct k (oddsOnlySieve limit) with
  | case1 i count hge =>
      have hi_size : i < (oddsOnlySieve limit).size :=
        sieve_index_lt_of_oddAt_le (le_trans hcandidate hbound)
      omega
  | case2 i hlt hprimeBool =>
      rw [scanNthPrimeIndex.eq_def]
      simp [hlt, hprimeBool]
      have hi_size : i < (oddsOnlySieve limit).size := Nat.lt_of_not_ge hlt
      have hp : Nat.Prime (oddAt i) :=
        (oddAt_pos_index_prime_iff hi_size hi_pos).1 hprimeBool
      have hck : Nat.count Nat.Prime (oddAt i) = k := by omega
      calc
        oddAt i = Nat.nth Nat.Prime (Nat.count Nat.Prime (oddAt i)) :=
          (Nat.nth_count hp).symm
        _ = Nat.nth Nat.Prime k := by rw [hck]
  | case3 i count hlt hprimeBool hcount_ne ih =>
      rw [scanNthPrimeIndex.eq_def]
      simp [hlt, hprimeBool, hcount_ne]
      have hi_size : i < (oddsOnlySieve limit).size := by omega
      have hp : Nat.Prime (oddAt i) :=
        (oddAt_pos_index_prime_iff hi_size hi_pos).1 hprimeBool
      have hlt_nth : oddAt i < Nat.nth Nat.Prime k := by
        by_contra hnot
        have heq : oddAt i = Nat.nth Nat.Prime k := by omega
        have hck : count = k := by
          rw [hcount, heq]
          exact Nat.count_nth_of_infinite Nat.infinite_setOf_prime k
        exact hcount_ne hck
      apply ih
      · rw [hcount, count_prime_oddAt_succ_of_prime hi_pos hp]
      · omega
      · rw [oddAt_succ]
        have hne : Nat.nth Nat.Prime k ≠ oddAt i + 1 := by
          intro heq
          exact not_prime_oddAt_add_one hi_pos (heq ▸ Nat.prime_nth_prime k)
        omega
  | case4 i count hlt hprimeBool ih =>
      rw [scanNthPrimeIndex.eq_def]
      simp [hlt, hprimeBool]
      have hi_size : i < (oddsOnlySieve limit).size := Nat.lt_of_not_ge hlt
      have hnp : ¬ Nat.Prime (oddAt i) := by
        intro hp
        exact hprimeBool ((oddAt_pos_index_prime_iff hi_size hi_pos).2 hp)
      have hlt_nth : oddAt i < Nat.nth Nat.Prime k := by
        by_contra hnot
        have heq : oddAt i = Nat.nth Nat.Prime k := by omega
        exact hnp (heq.symm ▸ Nat.prime_nth_prime k)
      apply ih
      · rw [hcount, count_prime_oddAt_succ_of_not_prime hi_pos hnp]
      · omega
      · rw [oddAt_succ]
        have hne : Nat.nth Nat.Prime k ≠ oddAt i + 1 := by
          intro heq
          exact not_prime_oddAt_add_one hi_pos (heq ▸ Nat.prime_nth_prime k)
        omega

lemma scanNthPrimeIndex_sound (limit k i count p : Nat)
    (hcount : count = Nat.count Nat.Prime (oddAt i))
    (hi_pos : 1 ≤ i)
    (hres : scanNthPrimeIndex k i count (oddsOnlySieve limit) = some p) :
    p = Nat.nth Nat.Prime k := by
  induction i, count using scanNthPrimeIndex.induct k (oddsOnlySieve limit) with
  | case1 i count hge =>
      rw [scanNthPrimeIndex.eq_def] at hres
      simp [hge] at hres
  | case2 i hlt hprimeBool =>
      rw [scanNthPrimeIndex.eq_def] at hres
      simp [hlt, hprimeBool] at hres
      subst p
      have hi_size : i < (oddsOnlySieve limit).size := Nat.lt_of_not_ge hlt
      have hp : Nat.Prime (oddAt i) :=
        (oddAt_pos_index_prime_iff hi_size hi_pos).1 hprimeBool
      have hck : Nat.count Nat.Prime (oddAt i) = k := by omega
      calc
        oddAt i = Nat.nth Nat.Prime (Nat.count Nat.Prime (oddAt i)) :=
          (Nat.nth_count hp).symm
        _ = Nat.nth Nat.Prime k := by rw [hck]
  | case3 i count hlt hprimeBool hcount_ne ih =>
      rw [scanNthPrimeIndex.eq_def] at hres
      simp [hlt, hprimeBool, hcount_ne] at hres
      have hi_size : i < (oddsOnlySieve limit).size := Nat.lt_of_not_ge hlt
      have hp : Nat.Prime (oddAt i) :=
        (oddAt_pos_index_prime_iff hi_size hi_pos).1 hprimeBool
      exact ih
        (by rw [hcount, count_prime_oddAt_succ_of_prime hi_pos hp])
        (by omega)
        hres
  | case4 i count hlt hprimeBool ih =>
      rw [scanNthPrimeIndex.eq_def] at hres
      simp [hlt, hprimeBool] at hres
      have hi_size : i < (oddsOnlySieve limit).size := Nat.lt_of_not_ge hlt
      have hnp : ¬ Nat.Prime (oddAt i) := by
        intro hp
        exact hprimeBool ((oddAt_pos_index_prime_iff hi_size hi_pos).2 hp)
      exact ih
        (by rw [hcount, count_prime_oddAt_succ_of_not_prime hi_pos hnp])
        (by omega)
        hres

lemma nthPrimeWithLimit_eq_nth (k limit : Nat)
    (hbound : Nat.nth Nat.Prime k ≤ limit) :
    nthPrimeWithLimit k limit = some (Nat.nth Nat.Prime k) := by
  cases k with
  | zero =>
      simp [nthPrimeWithLimit, Nat.nth_prime_zero_eq_two]
  | succ k =>
      unfold nthPrimeWithLimit
      exact scanNthPrimeIndex_eq_some limit (k + 1) 1 1 hbound
        (by native_decide)
        (by norm_num)
        (by
          have hle := Nat.add_two_le_nth_prime (k + 1)
          unfold oddAt
          omega)

lemma nthPrimeWithLimit_sound {k limit p : Nat}
    (hres : nthPrimeWithLimit k limit = some p) :
    p = Nat.nth Nat.Prime k := by
  cases k with
  | zero =>
      simp [nthPrimeWithLimit] at hres
      rw [Nat.nth_prime_zero_eq_two]
      exact hres.symm
  | succ k =>
      unfold nthPrimeWithLimit at hres
      exact scanNthPrimeIndex_sound limit (k + 1) 1 1 p
        (by native_decide) (by norm_num) hres

lemma nthPrimeWithRetry_eq_nth (k limit fuel : Nat)
    (hbound : Nat.nth Nat.Prime k ≤ limit * 2 ^ fuel) :
    nthPrimeWithRetry k limit fuel = Nat.nth Nat.Prime k := by
  induction fuel generalizing limit with
  | zero =>
      unfold nthPrimeWithRetry
      cases hres : nthPrimeWithLimit k limit with
      | some p =>
          exact nthPrimeWithLimit_sound hres
      | none =>
          have hlimit : Nat.nth Nat.Prime k ≤ limit := by
            simpa using hbound
          have hsome := nthPrimeWithLimit_eq_nth k limit hlimit
          rw [hres] at hsome
          contradiction
  | succ fuel ih =>
      unfold nthPrimeWithRetry
      cases hres : nthPrimeWithLimit k limit with
      | some p =>
          exact nthPrimeWithLimit_sound hres
      | none =>
          apply ih
          calc
            Nat.nth Nat.Prime k ≤ limit * 2 ^ (fuel + 1) := hbound
            _ = (limit * 2) * 2 ^ fuel := by
              rw [pow_succ]
              ring

theorem equiv (n : Nat) : ProjectEulerStatements.P7.naive n = solve n := by
  cases n with
  | zero => simp [ProjectEulerStatements.P7.naive, solve]
  | succ k =>
      unfold ProjectEulerStatements.P7.naive solve
      have hbound : Nat.nth Nat.Prime k ≤ initialLimit (k + 1) * 2 ^ (k + 1) := by
        calc
          Nat.nth Nat.Prime k ≤ 2 ^ (k + 1) := nth_prime_le_two_pow_succ k
          _ = 1 * 2 ^ (k + 1) := by simp
          _ ≤ initialLimit (k + 1) * 2 ^ (k + 1) :=
            Nat.mul_le_mul_right _ (one_le_initialLimit (k + 1))
      exact (nthPrimeWithRetry_eq_nth k (initialLimit (k + 1)) (k + 1)
        hbound).symm

end ProjectEulerSolutions.P7
