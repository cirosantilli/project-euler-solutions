import lang.py.builtin.type.int

namespace ProjectEulerSolutions.MillerRabin

/-- The source check performs at most this many squarings and returns on its first hit. -/
def squareSearch (n : Nat) : Nat → Int → Bool
  | 0, _ => false
  | fuel + 1, x =>
      let next := x * x % (n : Int)
      if next = (n : Int) - 1 then true else squareSearch n fuel next

/-- Exact algorithmic semantics of the lifted source helper, without a primality claim. -/
def millerCheck (a : Int) (n d s : Nat) : Bool :=
  if a % (n : Int) = 0 then true else
    let x := FormalSW.Py.Builtin.Int.powModBinary a d n
    if x = 1 ∨ x = (n : Int) - 1 then true else squareSearch n (s - 1) x


/-- Structural fuel bounds the original halving loop; actual work stops as soon as d is odd. -/
def decomposeAux : Nat → Nat → Nat → Nat × Nat
  | 0, d, s => (d, s)
  | fuel + 1, d, s =>
      if d % 2 = 0 then decomposeAux fuel (d / 2) (s + 1) else (d, s)

/-- The source starts the decomposition with d=n-1 and s=0. -/
def decompose (d : Nat) : Nat × Nat := decomposeAux d d 0

/-- Sufficient structural fuel produces the positive odd part and the exact power of two. -/
theorem decomposeAux_spec (fuel d s : Nat) (positive : 0 < d) (bound : d ≤ fuel) :
    0 < (decomposeAux fuel d s).1 ∧ (decomposeAux fuel d s).1 % 2 = 1 ∧
      (decomposeAux fuel d s).1 * 2 ^ (decomposeAux fuel d s).2 = d * 2 ^ s := by
  induction fuel generalizing d s with
  | zero => omega
  | succ fuel ih =>
      by_cases even : d % 2 = 0
      · rw [decomposeAux, if_pos even]
        have halfPositive : 0 < d / 2 := by omega
        have halfBound : d / 2 ≤ fuel := by omega
        obtain ⟨oddPositive, odd, identity⟩ := ih (d / 2) (s + 1) halfPositive halfBound
        refine ⟨oddPositive, odd, identity.trans ?_⟩
        have half : d / 2 * 2 = d := by omega
        calc
          d / 2 * 2 ^ (s + 1) = (d / 2 * 2) * 2 ^ s := by rw [Nat.pow_succ]; ac_rfl
          _ = d * 2 ^ s := by rw [half]
      · rw [decomposeAux, if_neg even]
        exact ⟨positive, by omega, rfl⟩

theorem decompose_spec (d : Nat) (positive : 0 < d) :
    0 < (decompose d).1 ∧ (decompose d).1 % 2 = 1 ∧
      (decompose d).1 * 2 ^ (decompose d).2 = d := by
  simpa [decompose] using decomposeAux_spec d d 0 positive (Nat.le_refl d)

/-- These are the exact source sequences, preserving small-prime equality early returns. -/
def smallPrimes : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
def bases : List Int := [2, 325, 9375, 28178, 450775, 9780504, 1795265022]

def smallScan (n : Int) : List Nat → Option Bool
  | [] => none
  | p :: ps =>
      if n = (p : Int) then some true else
        if n % (p : Int) = 0 then some false else smallScan n ps

/-- A source model, whose equality with genuine primality is a separate finite certificate obligation. -/
def isPrimeModel (n : Nat) : Bool :=
  if n < 2 then false else
    match smallScan n smallPrimes with
    | some answer => answer
    | none =>
        let ds := decompose (n - 1)
        bases.all (fun a => millerCheck a n ds.1 ds.2)

end ProjectEulerSolutions.MillerRabin
