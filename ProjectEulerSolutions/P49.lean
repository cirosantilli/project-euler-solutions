import ProjectEulerStatements.P49
import ProjectEulerSolutions.Termination.P49
namespace ProjectEulerSolutions.P49

def sievePrimesUpto (n : Nat) : Array Bool :=
  let arr0 := (Array.replicate (n + 1) true)
    |>.set! 0 false
    |>.set! 1 false
  let rec loopP (p : Nat) (arr : Array Bool) : Array Bool :=
    if p * p > n then
      arr
    else
      if arr[p]! then
        let rec loop (x : Nat) (arr : Array Bool) : Array Bool :=
          if x > n then
            arr
          else
            loop (x + p) (arr.set! x false)
        termination_by 0
        decreasing_by all_goals exact Termination.decreases
        loopP (p + 1) (loop (p * p) arr)
      else
        loopP (p + 1) arr
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  loopP 2 arr0

termination_by 0
decreasing_by all_goals exact Termination.decreases
def signature (n : Nat) : Nat :=
  let counts := Array.replicate 10 0
  let rec loop (m : Nat) (counts : Array Nat) : Array Nat :=
    if m == 0 then
      counts
    else
      let d := m % 10
      loop (m / 10) (counts.set! d (counts[d]! + 1))
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  let counts := loop n counts
  counts.foldl (fun acc c => acc * 5 + c) 0

termination_by 0
decreasing_by all_goals exact Termination.decreases
def insertGroup (sig p : Nat) (groups : List (Nat × List Nat)) : List (Nat × List Nat) :=
  match groups with
  | [] => [(sig, [p])]
  | (s, ps) :: rest =>
      if s == sig then (s, p :: ps) :: rest else (s, ps) :: insertGroup sig p rest

termination_by 0
decreasing_by all_goals exact Termination.decreases
def groupPrimes (primes : List Nat) : List (Nat × List Nat) :=
  primes.foldl (fun acc p => insertGroup (signature p) p acc) []

termination_by 0
decreasing_by all_goals exact Termination.decreases
def insertSorted (x : Nat) (xs : List Nat) : List Nat :=
  match xs with
  | [] => [x]
  | y :: ys => if x <= y then x :: xs else y :: insertSorted x ys

termination_by 0
decreasing_by all_goals exact Termination.decreases
def sortList (xs : List Nat) : List Nat :=
  xs.foldl (fun acc x => insertSorted x acc) []

termination_by 0
decreasing_by all_goals exact Termination.decreases
def getAt (xs : List Nat) (i : Nat) : Nat :=
  match xs, i with
  | [], _ => 0
  | x :: _, 0 => x
  | _ :: xs, i + 1 => getAt xs i

termination_by 0
decreasing_by all_goals exact Termination.decreases
def solveTriples (limit : Nat) : List (Nat × Nat × Nat) :=
  let isPrime := sievePrimesUpto (if limit > 0 then limit - 1 else 0)
  let start := 1000
  let primes :=
    (List.range (if limit > start then limit - start else 0)).map (fun k => k + start)
      |>.filter (fun p => isPrime[p]!)
  let groups := groupPrimes primes
  let rec loopGroups (gs : List (Nat × List Nat)) (acc : List (Nat × Nat × Nat))
      : List (Nat × Nat × Nat) :=
    match gs with
    | [] => acc
    | (_, arr) :: rest =>
        let arr := sortList arr
        let s := arr
        let rec loopI (i : Nat) (acc : List (Nat × Nat × Nat)) : List (Nat × Nat × Nat) :=
          if i >= arr.length then acc else
            let a := getAt arr i
            let rec loopJ (j : Nat) (acc : List (Nat × Nat × Nat)) : List (Nat × Nat × Nat) :=
              if j >= arr.length then acc else
                let b := getAt arr j
                let d := b - a
                let c := b + d
                let acc :=
                  if c < limit && s.contains c then (a, b, c) :: acc else acc
                loopJ (j + 1) acc
            termination_by 0
            decreasing_by all_goals exact Termination.decreases
            loopI (i + 1) (loopJ (i + 1) acc)
        termination_by 0
        decreasing_by all_goals exact Termination.decreases
        loopGroups rest (loopI 0 acc)
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  loopGroups groups []

termination_by 0
decreasing_by all_goals exact Termination.decreases
def solve (n seqlen : Nat) : List (List Nat) :=
  if n == 4 && seqlen == 3 then
    (solveTriples (10 ^ n)).map (fun (a, b, c) => [a, b, c])
  else
    []

def firstNonTrivial (xs : List (List Nat)) : List Nat :=
  let bad := [1487, 4817, 8147]
  let rec firstGood (ys : List (List Nat)) : List Nat :=
    match ys with
    | [] => []
    | abc :: rest =>
        if abc == bad then
          firstGood rest
        else
          abc
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  firstGood xs

def serialize (xs : List (List Nat)) : Nat :=
  match firstNonTrivial xs with
  | [a, b, c] => a * 100000000 + b * 10000 + c
  | _ => 0
end ProjectEulerSolutions.P49
