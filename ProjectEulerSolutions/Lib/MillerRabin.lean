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

end ProjectEulerSolutions.MillerRabin
