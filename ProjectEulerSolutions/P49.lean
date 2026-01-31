import ProjectEulerStatements.P49
namespace ProjectEulerSolutions.P49

partial def sievePrimesUpto (n : Nat) : Array Bool :=
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
        loopP (p + 1) (loop (p * p) arr)
      else
        loopP (p + 1) arr
  loopP 2 arr0

partial def signature (n : Nat) : Nat :=
  let counts := Array.replicate 10 0
  let rec loop (m : Nat) (counts : Array Nat) : Array Nat :=
    if m == 0 then
      counts
    else
      let d := m % 10
      loop (m / 10) (counts.set! d (counts[d]! + 1))
  let counts := loop n counts
  counts.foldl (fun acc c => acc * 5 + c) 0

partial def insertGroup (sig p : Nat) (groups : List (Nat × List Nat)) : List (Nat × List Nat) :=
  match groups with
  | [] => [(sig, [p])]
  | (s, ps) :: rest =>
      if s == sig then (s, p :: ps) :: rest else (s, ps) :: insertGroup sig p rest

partial def groupPrimes (primes : List Nat) : List (Nat × List Nat) :=
  primes.foldl (fun acc p => insertGroup (signature p) p acc) []

partial def insertSorted (x : Nat) (xs : List Nat) : List Nat :=
  match xs with
  | [] => [x]
  | y :: ys => if x <= y then x :: xs else y :: insertSorted x ys

partial def sortList (xs : List Nat) : List Nat :=
  xs.foldl (fun acc x => insertSorted x acc) []

partial def getAt (xs : List Nat) (i : Nat) : Nat :=
  match xs, i with
  | [], _ => 0
  | x :: _, 0 => x
  | _ :: xs, i + 1 => getAt xs i

partial def solve (limit : Nat) : List (Nat × Nat × Nat) :=
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
            loopI (i + 1) (loopJ (i + 1) acc)
        loopGroups rest (loopI 0 acc)
  loopGroups groups []

def firstNonTrivial (xs : List (Nat × Nat × Nat)) : List Nat :=
  let bad := (1487, 4817, 8147)
  let rec firstGood (ys : List (Nat × Nat × Nat)) : List Nat :=
    match ys with
    | [] => []
    | (a, b, c) :: rest =>
        if (a, b, c) == bad then
          firstGood rest
        else
          [a, b, c]
  firstGood xs

def serialize (xs : List (Nat × Nat × Nat)) : Nat :=
  match firstNonTrivial xs with
  | [a, b, c] => a * 1000000 + b * 1000 + c
  | _ => 0

theorem equiv (n : Nat) : ProjectEulerStatements.P49.naive n = solve n := sorry
end ProjectEulerSolutions.P49
