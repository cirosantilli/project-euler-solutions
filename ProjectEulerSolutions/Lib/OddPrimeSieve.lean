import ProjectEulerSolutions.Lib.PrimeSieve

/-! The odd-stride sieve first removes even composites, then visits only odd candidates. -/
namespace ProjectEulerSolutions.OddPrimeSieve

open PrimeSieve

def zeroRange (bytes : List UInt8) (start stop step : Nat) : List UInt8 :=
  (List.range' start ((stop - start + step - 1) / step) step).foldl
    (fun result i => result.set i 0) bytes

def initialEven (limit : Nat) : List UInt8 :=
  zeroRange (initial limit) 4 (limit + 1) 2

def oddMarkedIndices (length p : Nat) : List Nat :=
  List.range' (p * p) ((length - p * p + 2 * p - 1) / (2 * p)) (2 * p)

def oddMark (bytes : List UInt8) (p : Nat) : List UInt8 :=
  zeroRange bytes (p * p) bytes.length (2 * p)

theorem zeroRange_length (bytes : List UInt8) (start stop step : Nat) :
    (zeroRange bytes start stop step).length = bytes.length := by
  unfold zeroRange
  generalize List.range' start ((stop - start + step - 1) / step) step = indices
  induction indices generalizing bytes with
  | nil => rfl
  | cons i rest ih => simpa only [List.foldl_cons, List.length_set] using ih (bytes.set i 0)

theorem oddMark_length (bytes : List UInt8) (p : Nat) :
    (oddMark bytes p).length = bytes.length := zeroRange_length ..

theorem initialEven_length (limit : Nat) (large : 2 ≤ limit) :
    (initialEven limit).length = limit + 1 := by
  rw [initialEven, zeroRange_length]
  exact (sieved_initial limit large).1

theorem initialEven_eq_mark (limit : Nat) (large : 2 ≤ limit) :
    initialEven limit = mark (initial limit) 2 := by
  have length := (sieved_initial limit large).1
  simp [initialEven, zeroRange, mark, markedIndices, length]

theorem sieved_initialEven (limit : Nat) (large : 2 ≤ limit) :
    Sieved limit 3 (initialEven limit) := by
  rw [initialEven_eq_mark limit large]
  exact sieved_mark limit 2 _ (sieved_initial limit large) Nat.prime_two

theorem mem_oddMarkedIndices (length p i : Nat) (positive : 0 < p) :
    i ∈ oddMarkedIndices length p ↔ p * p ≤ i ∧ i < length ∧ 2 * p ∣ i - p * p := by
  unfold oddMarkedIndices
  exact FormalSW.Py.Builtin.Slice.mem_positive_end _ _ _ _ (by omega)

theorem oddMark_get (bytes : List UInt8) (p i : Nat) (positive : 0 < p)
    (valid : i < bytes.length) :
    (oddMark bytes p)[i]? =
      if p * p ≤ i ∧ 2 * p ∣ i - p * p then some 0 else bytes[i]? := by
  rw [oddMark, zeroRange, FormalSW.Py.Builtin.List.getElem?_setUniform _ _ _ _ valid]
  change (if i ∈ oddMarkedIndices bytes.length p then some 0 else bytes[i]?) = _
  simp only [mem_oddMarkedIndices _ _ _ positive, valid, true_and]

theorem odd_stride_divides (p i : Nat) (oddP : p % 2 = 1) (oddI : i % 2 = 1)
    (lower : p * p ≤ i) : 2 * p ∣ i - p * p ↔ p ∣ i := by
  constructor
  · intro divides
    have pd : p ∣ i - p * p := dvd_trans (dvd_mul_left p 2) divides
    exact (Nat.dvd_sub_iff_left lower (dvd_mul_right p p)).mp pd
  · intro divides
    obtain ⟨k, hk⟩ := (Nat.dvd_sub_iff_left lower (dvd_mul_right p p)).mpr divides
    have equation : i = p * p + p * k := by omega
    have evenK : k % 2 = 0 := by
      have parity := congrArg (fun n : Nat => n % 2) equation
      simp [Nat.add_mod, Nat.mul_mod, oddP, oddI] at parity
      omega
    obtain ⟨j, hj⟩ := Nat.dvd_of_mod_eq_zero evenK
    refine ⟨j, ?_⟩
    rw [hk, hj]
    ring

theorem survives_large_odd (p i : Nat) (lower : 3 ≤ p) (large : 4 ≤ i)
    (survives : Survives p i) : i % 2 = 1 := by
  have notEven := survives 2 (by omega) Nat.prime_two (by omega)
  have : i % 2 ≠ 0 := fun h => notEven (Nat.dvd_of_mod_eq_zero h)
  omega

theorem next_composite (p : Nat) (lower : 3 ≤ p) (oddP : p % 2 = 1) :
    ¬ Nat.Prime (p + 1) := by
  intro prime
  rcases prime.eq_two_or_odd with small | oddNext <;> omega

theorem sieved_oddSkip (limit p : Nat) (bytes : List UInt8)
    (sieved : Sieved limit p bytes) (lower : 3 ≤ p) (oddP : p % 2 = 1)
    (composite : ¬ Nat.Prime p) : Sieved limit (p + 2) bytes := by
  exact sieved_skip limit (p + 1) bytes (sieved_skip limit p bytes sieved composite)
    (next_composite p lower oddP)

theorem sieved_oddMark (limit p : Nat) (bytes : List UInt8)
    (sieved : Sieved limit p bytes) (lower : 3 ≤ p) (prime : Nat.Prime p) :
    Sieved limit (p + 2) (oddMark bytes p) := by
  have oddP : p % 2 = 1 := prime.eq_two_or_odd.resolve_left (by omega)
  apply sieved_skip limit (p + 1) _ _ (next_composite p lower oddP)
  refine ⟨by rw [oddMark_length, sieved.1], ?_⟩
  intro i byte lowerI found
  have valid : i < bytes.length := by
    have := (List.getElem?_eq_some_iff.mp found).1
    simpa only [oddMark_length] using this
  rw [oddMark_get bytes p i prime.pos valid] at found
  rw [survives_next]
  by_cases removed : p * p ≤ i ∧ 2 * p ∣ i - p * p
  · simp only [if_pos removed, Option.some.injEq] at found
    subst byte
    have divides : p ∣ i := (Nat.dvd_sub_iff_left removed.1 (dvd_mul_right p p)).mp
      (dvd_trans (dvd_mul_left p 2) removed.2)
    simp [prime, divides, show p ^ 2 ≤ i by nlinarith [removed.1]]
  · simp only [if_neg removed] at found
    rw [sieved.2 i byte lowerI found]
    refine ⟨fun survives => ⟨survives, ?_⟩, And.left⟩
    intro _ square divides
    have oddI := survives_large_odd p i lower (by nlinarith) survives
    exact removed ⟨by nlinarith, (odd_stride_divides p i oddP oddI (by nlinarith)).mpr divides⟩

/-- Sieve cells retain Python's exact zero/one byte representation. -/
def Binary (bytes : List UInt8) : Prop := ∀ byte ∈ bytes, byte = 0 ∨ byte = 1

theorem binary_set_zero (bytes : List UInt8) (i : Nat) (binary : Binary bytes) :
    Binary (bytes.set i 0) := by
  intro byte member
  rcases List.mem_or_eq_of_mem_set member with old | zero
  · exact binary byte old
  · exact Or.inl zero

theorem zeroRange_binary (bytes : List UInt8) (start stop step : Nat)
    (binary : Binary bytes) : Binary (zeroRange bytes start stop step) := by
  unfold zeroRange
  generalize List.range' start ((stop - start + step - 1) / step) step = indices
  induction indices generalizing bytes with
  | nil => exact binary
  | cons i rest ih => exact ih (bytes.set i 0) (binary_set_zero bytes i binary)

theorem initialEven_binary (limit : Nat) : Binary (initialEven limit) := by
  apply zeroRange_binary
  intro byte member
  simp only [initial, List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with (zero | zero) | rest
  · exact Or.inl zero
  · exact Or.inl zero
  · exact Or.inr (List.eq_of_mem_replicate rest)

theorem oddMark_binary (bytes : List UInt8) (p : Nat) (binary : Binary bytes) :
    Binary (oddMark bytes p) := zeroRange_binary bytes _ _ _ binary

theorem zeroRange_get_before (bytes : List UInt8) (start stop step i : Nat)
    (before : i < start) (positive : 0 < step) :
    (zeroRange bytes start stop step)[i]? = bytes[i]? := by
  by_cases valid : i < bytes.length
  · rw [zeroRange, FormalSW.Py.Builtin.List.getElem?_setUniform _ _ _ _ valid]
    have missing : i ∉ List.range' start ((stop - start + step - 1) / step) step := by
      rw [FormalSW.Py.Builtin.Slice.mem_positive_end _ _ _ _ positive]
      omega
    simp only [missing, if_false]
  · rw [List.getElem?_eq_none (by simpa only [zeroRange_length] using Nat.le_of_not_gt valid),
      List.getElem?_eq_none (Nat.le_of_not_gt valid)]

theorem initialEven_get_low (limit i : Nat) (small : i < 2) :
    (initialEven limit)[i]? = some 0 := by
  rw [initialEven, zeroRange_get_before _ _ _ _ _ (by omega) (by omega)]
  have cases : i = 0 ∨ i = 1 := by omega
  rcases cases with rfl | rfl <;> simp [initial]

theorem oddMark_get_low (bytes : List UInt8) (p i : Nat) (lower : 3 ≤ p) (small : i < 2) :
    (oddMark bytes p)[i]? = bytes[i]? := by
  exact zeroRange_get_before bytes _ _ _ _ (by nlinarith) (by omega)

theorem sieved_finished_get (limit p i : Nat) (bytes : List UInt8)
    (sieved : Sieved limit p bytes) (binary : Binary bytes)
    (zero : ∀ i < 2, bytes[i]? = some 0) (finished : limit < p ^ 2) (valid : i ≤ limit) :
    bytes[i]? = some (if Nat.Prime i then 1 else 0) := by
  have inside : i < bytes.length := by rw [sieved.1]; omega
  obtain ⟨byte, found⟩ : ∃ byte, bytes[i]? = some byte :=
    ⟨bytes[i], List.getElem?_eq_getElem inside⟩
  by_cases small : i < 2
  · rw [zero i small, if_neg (fun prime => by have := prime.two_le; omega)]
  · have correct := (sieved.2 i byte (by omega) found).trans
      (survives_finished limit p i (by omega) valid finished)
    rcases binary byte (List.mem_of_getElem? found) with empty | full
    · subst byte
      have composite : ¬ Nat.Prime i := by simpa using correct.symm
      simpa [composite] using found
    · subst byte
      have prime : Nat.Prime i := correct.mp (by decide)
      simpa [prime] using found

theorem zeroRange_bounds (start stop step : Nat) (positive : 0 < step) :
    let count := (stop - start + step - 1) / step
    stop ≤ start + count * step ∧
      (count = 0 ∨ start + (count - 1) * step < stop) := by
  dsimp
  by_cases within : start < stop
  · have count : (stop - start + step - 1) / step = (stop - start - 1) / step + 1 := by
      rw [show stop - start + step - 1 = (stop - start - 1) + step by omega,
        Nat.add_div_right _ positive]
    rw [count]
    have quotient := Nat.div_mul_le_self (stop - start - 1) step
    have remainder := Nat.mod_add_div (stop - start - 1) step
    have bound := Nat.mod_lt (stop - start - 1) positive
    constructor
    · nlinarith [show start + (stop - start - 1) + 1 = stop by omega]
    · right
      simp only [Nat.add_sub_cancel]
      omega
  · have empty : stop - start = 0 := by omega
    simp only [empty, Nat.zero_add,
      Nat.div_eq_of_lt (show step - 1 < step by omega), Nat.zero_mul, Nat.add_zero]
    exact ⟨by omega, Or.inl trivial⟩

end ProjectEulerSolutions.OddPrimeSieve
