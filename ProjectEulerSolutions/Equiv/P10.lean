import ProjectEulerSolutions.P10
import Mathlib.Tactic
namespace ProjectEulerSolutions.P10

open scoped BigOperators

lemma naive_succ (n : Nat) :
    ProjectEulerStatements.P10.naive (n + 1) =
      ProjectEulerStatements.P10.naive n + if Nat.Prime n then n else 0 := by
  unfold ProjectEulerStatements.P10.naive
  rw [Finset.sum_filter, Finset.sum_filter]
  rw [Finset.sum_range_succ]

lemma naive_zero : ProjectEulerStatements.P10.naive 0 = 0 := by
  simp [ProjectEulerStatements.P10.naive]

lemma naive_one : ProjectEulerStatements.P10.naive 1 = 0 := by
  rw [show 1 = 0 + 1 by rfl, naive_succ, naive_zero]
  norm_num

lemma naive_two : ProjectEulerStatements.P10.naive 2 = 0 := by
  rw [show 2 = 1 + 1 by rfl, naive_succ, naive_one]
  norm_num

lemma naive_three : ProjectEulerStatements.P10.naive 3 = 2 := by
  rw [show 3 = 2 + 1 by rfl, naive_succ, naive_two]
  norm_num [Nat.prime_two]

lemma solve_of_le_two {n : Nat} (h : n ≤ 2) : solve n = 0 := by
  unfold solve
  simp [h]

lemma solve_of_eq_three : solve 3 = 2 := by
  unfold solve
  simp

lemma equiv_of_le_two {n : Nat} (h : n ≤ 2) : ProjectEulerStatements.P10.naive n = solve n := by
  interval_cases n <;> simp [naive_zero, naive_one, naive_two, solve]

lemma equiv_three : ProjectEulerStatements.P10.naive 3 = solve 3 := by
  rw [naive_three, solve_of_eq_three]

lemma not_prime_even_gt_two {n : Nat} (h2 : 2 < n) (heven : Even n) : ¬ Nat.Prime n := by
  intro hp
  have hodd := hp.odd_of_ne_two (by omega)
  exact Nat.not_even_iff_odd.mpr hodd heven

lemma isPrimeWithLoop_eq_false_of_current_dvd {n : Nat} {primes : Array Nat} {i : Nat}
    (hi : i < primes.size) (hle : primes[i]! * primes[i]! ≤ n) (hdvd : primes[i]! ∣ n) :
    isPrimeWithLoop n primes i = false := by
  rw [isPrimeWithLoop.eq_def]
  have hnotge : ¬ i ≥ primes.size := by omega
  have hnotgt : ¬ primes[i]! * primes[i]! > n := by omega
  have hmod : (n % primes[i]! == 0) = true := by
    rw [Nat.mod_eq_zero_of_dvd hdvd]
    decide
  simp [hnotge, hnotgt, hmod]

lemma isPrimeWithLoop_eq_true_of_prime {n : Nat} {primes : Array Nat} {i : Nat}
    (hn : Nat.Prime n)
    (hprime : ∀ j, i ≤ j → j < primes.size → Nat.Prime primes[j]!)
    (hlt : ∀ j, i ≤ j → j < primes.size → primes[j]! < n) :
    isPrimeWithLoop n primes i = true := by
  induction i using isPrimeWithLoop.induct n primes with
  | case1 x hx =>
      rw [isPrimeWithLoop.eq_def]
      simp [hx]
  | case2 x hx p hpgt =>
      rw [isPrimeWithLoop.eq_def]
      simp [hx]
      left
      simpa [p] using hpgt
  | case3 x hx p _hpgt hmod =>
      have hxlt : x < primes.size := by omega
      have hp : Nat.Prime primes[x]! := hprime x (by omega) hxlt
      have hplt : primes[x]! < n := hlt x (by omega) hxlt
      have hdvd : primes[x]! ∣ n := by
        exact Nat.dvd_of_mod_eq_zero (by simpa [p] using hmod)
      have heq : n = primes[x]! := (Nat.Prime.dvd_iff_eq hn hp.ne_one).1 hdvd
      omega
  | case4 x hx p _hpgt hmod ih =>
      rw [isPrimeWithLoop.eq_def]
      simp [hx]
      right
      constructor
      · intro hz
        exact hmod (by simp [p, hz])
      · apply ih
        · intro j hj hjlt
          exact hprime j (by omega) hjlt
        · intro j hj hjlt
          exact hlt j (by omega) hjlt

lemma isPrimeWith_eq_true_of_prime {n : Nat} {primes : Array Nat}
    (hn : Nat.Prime n)
    (hprime : ∀ j, j < primes.size → Nat.Prime primes[j]!)
    (hlt : ∀ j, j < primes.size → primes[j]! < n) :
    isPrimeWith n primes = true := by
  unfold isPrimeWith
  exact isPrimeWithLoop_eq_true_of_prime hn
    (by intro j _ hjlt; exact hprime j hjlt)
    (by intro j _ hjlt; exact hlt j hjlt)

abbrev ArrayHas (primes : Array Nat) (p : Nat) : Prop :=
  ∃ i, i < primes.size ∧ primes[i]! = p

abbrev PrimeArrayOK (candidate : Nat) (primes : Array Nat) : Prop :=
  (∀ i, i < primes.size → Nat.Prime primes[i]! ∧ primes[i]! < candidate) ∧
  (∀ i j, i < j → j < primes.size → primes[i]! < primes[j]!) ∧
  (∀ p, Nat.Prime p → p < candidate → ArrayHas primes p)

lemma array_getElem!_push_lt {primes : Array Nat} {p : Nat} {i : Nat} (hi : i < primes.size) :
    (primes.push p)[i]! = primes[i]! := by
  have hpush : i < (primes.push p).size := by
    simpa using Nat.lt_trans hi (by simp : primes.size < (primes.push p).size)
  calc
    (primes.push p)[i]! = (primes.push p)[i] := getElem!_pos (primes.push p) i hpush
    _ = primes[i] := Array.getElem_push_lt hi
    _ = primes[i]! := (getElem!_pos primes i hi).symm

lemma array_getElem!_push_eq (primes : Array Nat) (p : Nat) :
    (primes.push p)[primes.size]! = p := by
  have hpush : primes.size < (primes.push p).size := by simp
  calc
    (primes.push p)[primes.size]! = (primes.push p)[primes.size] :=
      getElem!_pos (primes.push p) primes.size hpush
    _ = p := Array.getElem_push_eq

lemma array_has_push_self (primes : Array Nat) (p : Nat) : ArrayHas (primes.push p) p := by
  refine ⟨primes.size, ?_, ?_⟩
  · simp
  · exact array_getElem!_push_eq primes p

lemma array_has_push_of_has {primes : Array Nat} {p q : Nat} (h : ArrayHas primes p) :
    ArrayHas (primes.push q) p := by
  rcases h with ⟨i, hi, hp⟩
  refine ⟨i, ?_, ?_⟩
  · simpa using Nat.lt_trans hi (by simp : primes.size < (primes.push q).size)
  · rw [array_getElem!_push_lt hi, hp]

lemma primeArrayOK_initial : PrimeArrayOK 3 #[2] := by
  constructor
  · intro i hi
    have hi1 : i < 1 := by simpa using hi
    interval_cases i
    simp [Nat.prime_two]
  constructor
  · intro i j hij hj
    have hj1 : j < 1 := by simpa using hj
    interval_cases j
    omega
  · intro p hp hplt
    have hp2 : 2 ≤ p := hp.two_le
    have hp_eq : p = 2 := by omega
    subst p
    exact ⟨0, by simp, by simp⟩

lemma primeArrayOK_push {candidate : Nat} {primes : Array Nat}
    (hok : PrimeArrayOK candidate primes) (hodd : Odd candidate) (hprime : Nat.Prime candidate) :
    PrimeArrayOK (candidate + 2) (primes.push candidate) := by
  rcases hok with ⟨hall, hmono, hcomplete⟩
  constructor
  · intro i hi
    by_cases hlt : i < primes.size
    · rw [array_getElem!_push_lt hlt]
      exact ⟨(hall i hlt).1, Nat.lt_trans (hall i hlt).2 (by omega)⟩
    · have hieq : i = primes.size := by
        have hs : (primes.push candidate).size = primes.size + 1 := by simp
        omega
      subst i
      rw [array_getElem!_push_eq]
      exact ⟨hprime, by omega⟩
  constructor
  · intro i j hij hj
    by_cases hjold : j < primes.size
    · have hiold : i < primes.size := by omega
      rw [array_getElem!_push_lt hiold, array_getElem!_push_lt hjold]
      exact hmono i j hij hjold
    · have hjeq : j = primes.size := by
        have hs : (primes.push candidate).size = primes.size + 1 := by simp
        omega
      subst j
      have hiold : i < primes.size := by omega
      rw [array_getElem!_push_lt hiold, array_getElem!_push_eq]
      exact (hall i hiold).2
  · intro p hp hplt
    by_cases hp_lt_old : p < candidate
    · exact array_has_push_of_has (hcomplete p hp hp_lt_old)
    · have hp_eq_or : p = candidate ∨ p = candidate + 1 := by omega
      rcases hp_eq_or with hp_eq | hp_eq
      · subst p
        exact array_has_push_self primes candidate
      · exfalso
        have heven : Even p := by
          rw [hp_eq]
          rcases hodd with ⟨k, hk⟩
          use k + 1
          omega
        exact not_prime_even_gt_two (by have h2 := hprime.two_le; omega) heven hp

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

lemma isPrimeWithLoop_eq_false_of_dvd_later {n : Nat} {primes : Array Nat} {i j : Nat}
    (hmono : ∀ a b, a < b → b < primes.size → primes[a]! < primes[b]!)
    (hij : i ≤ j) (hj : j < primes.size) (hsq : primes[j]! * primes[j]! ≤ n)
    (hdvd : primes[j]! ∣ n) :
    isPrimeWithLoop n primes i = false := by
  induction i using isPrimeWithLoop.induct n primes with
  | case1 x hx =>
      omega
  | case2 x _hx p hpgt =>
      have hsqx : primes[x]! * primes[x]! ≤ n := by
        by_cases hxeq : x = j
        · subst x
          exact hsq
        · have hxltj : x < j := by omega
          have hltp : primes[x]! < primes[j]! := hmono x j hxltj hj
          exact le_trans (Nat.mul_le_mul (Nat.le_of_lt hltp) (Nat.le_of_lt hltp)) hsq
      have hsqxp : p * p ≤ n := by simpa [p] using hsqx
      exact False.elim ((not_lt_of_ge hsqxp) hpgt)
  | case3 x hx p hpgt hmod =>
      rw [isPrimeWithLoop.eq_def]
      simp [hx]
      have hsqx : primes[x]! * primes[x]! ≤ n := by simpa [p] using le_of_not_gt hpgt
      constructor
      · exact hsqx
      · intro hnot
        exact False.elim (hnot (by simpa [p] using hmod))
  | case4 x hx p hpgt hmod ih =>
      rw [isPrimeWithLoop.eq_def]
      simp [hx]
      have hsqx : primes[x]! * primes[x]! ≤ n := by simpa [p] using le_of_not_gt hpgt
      constructor
      · exact hsqx
      · intro _hnot
        have hxltj : x < j := by
          by_contra hnotlt
          have hxej : x = j := by omega
          subst x
          have hz : (n % primes[j]! == 0) = true := by
            rw [Nat.mod_eq_zero_of_dvd hdvd]
            decide
          exact hmod (by simpa [p] using hz)
        exact ih (by omega)

lemma isPrimeWith_eq_false_of_not_prime {candidate : Nat} {primes : Array Nat}
    (hok : PrimeArrayOK candidate primes) (hcand : 2 ≤ candidate) (hnp : ¬ Nat.Prime candidate) :
    isPrimeWith candidate primes = false := by
  rcases exists_prime_dvd_sq_le_of_not_prime hcand hnp with ⟨p, hp, hpdvd, hpsq⟩
  have hp_lt : p < candidate := by
    by_contra hge
    have hple : p ≤ candidate := Nat.le_of_dvd (by omega : 0 < candidate) hpdvd
    have hpeq : p = candidate := by omega
    subst p
    exact hnp hp
  rcases hok.2.2 p hp hp_lt with ⟨j, hj, hjp⟩
  unfold isPrimeWith
  apply isPrimeWithLoop_eq_false_of_dvd_later (n := candidate) (primes := primes) (i := 0) (j := j)
  · exact hok.2.1
  · omega
  · exact hj
  · simpa [hjp] using hpsq
  · simpa [hjp] using hpdvd

lemma naive_add_two_of_odd {candidate : Nat} (hc3 : 3 ≤ candidate) (hodd : Odd candidate) :
    ProjectEulerStatements.P10.naive (candidate + 2) =
      ProjectEulerStatements.P10.naive candidate + if Nat.Prime candidate then candidate else 0 := by
  rw [show candidate + 2 = (candidate + 1) + 1 by omega, naive_succ]
  rw [naive_succ]
  have hnp : ¬ Nat.Prime (candidate + 1) := by
    have heven : Even (candidate + 1) := by
      rcases hodd with ⟨k, hk⟩
      use k + 1
      omega
    exact not_prime_even_gt_two (by omega) heven
  simp [hnp]

lemma naive_eq_of_stop {limit candidate : Nat} (hge : candidate ≥ limit)
    (hle : candidate ≤ limit + 1) (hodd : Odd candidate) (hlimit3 : 3 < limit) :
    ProjectEulerStatements.P10.naive candidate = ProjectEulerStatements.P10.naive limit := by
  have hcases : candidate = limit ∨ candidate = limit + 1 := by omega
  rcases hcases with rfl | hsucc
  · rfl
  · subst candidate
    rw [naive_succ]
    have hnp : ¬ Nat.Prime limit := by
      have heven : Even limit := by
        rcases hodd with ⟨k, hk⟩
        use k
        omega
      exact not_prime_even_gt_two (by omega) heven
    simp [hnp]

lemma primeArrayOK_skip {candidate : Nat} {primes : Array Nat}
    (hok : PrimeArrayOK candidate primes) (hc3 : 3 ≤ candidate) (hodd : Odd candidate)
    (hnp : ¬ Nat.Prime candidate) :
    PrimeArrayOK (candidate + 2) primes := by
  rcases hok with ⟨hall, hmono, hcomplete⟩
  constructor
  · intro i hi
    exact ⟨(hall i hi).1, Nat.lt_trans (hall i hi).2 (by omega)⟩
  constructor
  · exact hmono
  · intro p hp hplt
    by_cases hp_lt_old : p < candidate
    · exact hcomplete p hp hp_lt_old
    · have hp_eq_or : p = candidate ∨ p = candidate + 1 := by omega
      rcases hp_eq_or with hp_eq | hp_eq
      · subst p
        exact (hnp hp).elim
      · exfalso
        have heven : Even p := by
          rw [hp_eq]
          rcases hodd with ⟨k, hk⟩
          use k + 1
          omega
        exact not_prime_even_gt_two (by omega) heven hp

lemma odd_add_two {n : Nat} (h : Odd n) : Odd (n + 2) := by
  rcases h with ⟨k, hk⟩
  use k + 1
  omega

lemma go_correct {limit candidate sum : Nat} {primes : Array Nat}
    (hlimit : 3 < limit) (hc3 : 3 ≤ candidate) (hodd : Odd candidate)
    (hle : candidate ≤ limit + 1)
    (hsum : sum = ProjectEulerStatements.P10.naive candidate)
    (hok : PrimeArrayOK candidate primes) :
    go limit candidate sum primes = ProjectEulerStatements.P10.naive limit := by
  induction candidate, sum, primes using go.induct limit with
  | case1 candidate sum primes hge =>
      rw [go.eq_def]
      simp [hge]
      rw [hsum]
      exact naive_eq_of_stop hge hle hodd hlimit
  | case2 candidate sum primes hnotge hisp ih =>
      have hprime : Nat.Prime candidate := by
        by_contra hnp
        have hfalse := isPrimeWith_eq_false_of_not_prime hok (by omega : 2 ≤ candidate) hnp
        rw [hfalse] at hisp
        contradiction
      rw [go.eq_def]
      simp [hnotge, hisp]
      apply ih
      · omega
      · exact odd_add_two hodd
      · omega
      · rw [hsum, naive_add_two_of_odd hc3 hodd]
        simp [hprime]
      · exact primeArrayOK_push hok hodd hprime
  | case3 candidate sum primes hnotge hisp ih =>
      have hnp : ¬ Nat.Prime candidate := by
        intro hp
        have htrue := isPrimeWith_eq_true_of_prime hp
          (by intro j hj; exact (hok.1 j hj).1)
          (by intro j hj; exact (hok.1 j hj).2)
        exact hisp htrue
      rw [go.eq_def]
      simp [hnotge, hisp]
      apply ih
      · omega
      · exact odd_add_two hodd
      · omega
      · rw [hsum, naive_add_two_of_odd hc3 hodd]
        simp [hnp]
      · exact primeArrayOK_skip hok hc3 hodd hnp

theorem equiv (n : Nat) : ProjectEulerStatements.P10.naive n = solve n := by
  by_cases h2 : n ≤ 2
  · exact equiv_of_le_two h2
  · by_cases h3 : n = 3
    · subst n
      exact equiv_three
    · have hgt : 3 < n := by omega
      unfold solve
      simp [show ¬ n ≤ 2 by omega, show ¬ n ≤ 3 by omega]
      rw [go_correct (limit := n) (candidate := 3) (sum := 2) (primes := #[2])]
      · exact hgt
      · norm_num
      · norm_num
      · omega
      · exact naive_three.symm
      · exact primeArrayOK_initial
end ProjectEulerSolutions.P10
