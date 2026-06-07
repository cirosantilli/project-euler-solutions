import ProjectEulerStatements.P7
namespace ProjectEulerSolutions.P7

def initialLimit (n : Nat) : Nat :=
  if n < 6 then
    15
  else
    n * (Nat.log2 n + Nat.log2 (Nat.log2 n + 1) + 1) + 10

def sieveSize (limit : Nat) : Nat :=
  limit / 2 + 1

def oddAt (i : Nat) : Nat :=
  2 * i + 1

def initialSieve (limit : Nat) : Array Bool :=
  Array.ofFn (fun i : Fin (sieveSize limit) => decide (i.val != 0))

def markedByIndex (i idx : Nat) : Prop :=
  1 ≤ i ∧ oddAt i * oddAt i ≤ oddAt idx ∧ (oddAt i ∣ oddAt idx)

instance (i idx : Nat) : Decidable (markedByIndex i idx) := by
  unfold markedByIndex
  infer_instance

def hasMarkedFactorBefore (i idx : Nat) : Prop :=
  ∃ j ∈ Finset.range i, markedByIndex j idx

instance (i idx : Nat) : Decidable (hasMarkedFactorBefore i idx) := by
  unfold hasMarkedFactorBefore
  infer_instance

def sieveState (limit processed : Nat) : Array Bool :=
  Array.ofFn (fun idx : Fin (sieveSize limit) =>
    decide (idx.val != 0 ∧ ¬ hasMarkedFactorBefore processed idx.val))

def markMultiples (p : Nat) (arr : Array Bool) : Array Bool :=
  Array.ofFn (fun i : Fin arr.size =>
    if p * p ≤ oddAt i.val ∧ oddAt i.val % p = 0 then false else arr[i.val]!)

def sieveLoop (limit maxI i : Nat) (isPrime : Array Bool) : Array Bool :=
  if i > maxI then
    isPrime
  else if isPrime[i]! then
    let p := oddAt i
    sieveLoop limit maxI (i + 1) (markMultiples p isPrime)
  else
    sieveLoop limit maxI (i + 1) isPrime
termination_by maxI + 1 - i
decreasing_by
  all_goals omega

def oddsOnlySieveImpl (limit : Nat) : Array Bool :=
  let isPrime := initialSieve limit
  let root := Nat.sqrt limit
  let maxI := root / 2
  sieveLoop limit maxI 1 isPrime

def oddsOnlySieve (limit : Nat) : Array Bool :=
  oddsOnlySieveImpl limit

def scanNthPrimeIndex (k i count : Nat) (isPrime : Array Bool) : Option Nat :=
  if h : i >= isPrime.size then
    none
  else
    let p := oddAt i
    if isPrime[i]! then
      if count = k then
        some p
      else
        scanNthPrimeIndex k (i + 1) (count + 1) isPrime
    else
      scanNthPrimeIndex k (i + 1) count isPrime
termination_by isPrime.size - i
decreasing_by
  all_goals omega

def nthPrimeWithLimit (k limit : Nat) : Option Nat :=
  match k with
  | 0 => some 2
  | _ + 1 =>
      let isPrime := oddsOnlySieve limit
      scanNthPrimeIndex k 1 1 isPrime

def nthPrimeWithRetry (k limit fuel : Nat) : Nat :=
  match nthPrimeWithLimit k limit with
  | some p => p
  | none =>
      match fuel with
      | 0 => 0
      | fuel + 1 => nthPrimeWithRetry k (limit * 2) fuel
termination_by fuel

def solve (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | k + 1 => nthPrimeWithRetry k (initialLimit (k + 1)) (k + 1)

example : solve 1 = 2 := by
  native_decide

example : solve 6 = 13 := by
  native_decide
end ProjectEulerSolutions.P7
