import ProjectEulerSolutions.Equiv.P7
import ProjectEulerStatements.Lib.PrimePartitions

/-! Inclusive prime lists obtained from the proved odd-only array sieve. -/
namespace ProjectEulerSolutions.PrimeList
set_option Elab.async false

def selected (flags : Array Bool) (n : Nat) : Bool :=
  if n = 2 then true else if n % 2 = 1 then flags.getD (n/2) false else false

theorem selected_correct (limit n : Nat) (within : n ≤ limit) :
    selected (P7.oddsOnlySieveImpl limit) n = decide (Nat.Prime n) := by
  by_cases two : n = 2
  · simp [selected, two, Nat.prime_two]
  · by_cases odd : n % 2 = 1
    · have inside : n/2 < (P7.oddsOnlySieveImpl limit).size := by
        rw [P7.oddsOnlySieveImpl_size, P7.sieveSize]
        omega
      have oddAt : P7.oddAt (n/2) = n := by unfold P7.oddAt; omega
      have read := P7.oddsOnlySieveImpl_get! inside
      have same : (P7.oddsOnlySieveImpl limit).getD (n/2) false = (P7.oddsOnlySieveImpl limit)[n/2]! := by
        simp [Array.getD, inside]
      rw [selected, if_neg two, if_pos odd, same, read, oddAt]
      have equivalent : (n/2 != 0 ∧ Nat.Prime n) ↔ Nat.Prime n := by
        constructor
        · exact And.right
        · intro prime
          have := prime.two_le
          exact ⟨by simp [show n/2 ≠ 0 by omega], prime⟩
      simp only [equivalent]
    · have composite : ¬ Nat.Prime n := by
        intro prime
        exact prime.eq_two_or_odd.elim two odd
      simp [selected, two, odd, composite]

def sievePrimes (limit : Nat) : List Nat :=
  let flags := P7.oddsOnlySieveImpl limit
  (List.range (limit+1)).filter (selected flags)

theorem sievePrimes_correct (limit : Nat) :
    sievePrimes limit = ProjectEulerStatements.PrimePartitions.primesUpTo limit := by
  apply List.filter_congr
  intro n member
  exact selected_correct limit n (by have := List.mem_range.mp member; omega)

end ProjectEulerSolutions.PrimeList
