import ProjectEulerStatements.P41
import ProjectEulerSolutions.Termination.P41
namespace ProjectEulerSolutions.P41

def isPrime (n : Nat) : Bool :=
  if n < 2 then
    false
  else if n % 2 == 0 then
    n == 2
  else if n % 3 == 0 then
    n == 3
  else
    let rec loop (i : Nat) : Bool :=
      if i * i > n then
        true
      else if n % i == 0 || n % (i + 2) == 0 then
        false
      else
        loop (i + 6)
    termination_by 0
    decreasing_by all_goals exact Termination.decreases
    loop 5

termination_by 0
decreasing_by all_goals exact Termination.decreases
def permToInt (perm : List Nat) : Nat :=
  perm.foldl (fun acc d => acc * 10 + d) 0

termination_by 0
decreasing_by all_goals exact Termination.decreases
def permutations (xs : List Nat) : List (List Nat) :=
  match xs with
  | [] => [[]]
  | _ =>
      let rec concatMap (f : List Nat -> List (List Nat)) (ls : List (List Nat)) : List (List Nat) :=
        match ls with
        | [] => []
        | x :: xs => f x ++ concatMap f xs
      termination_by 0
      decreasing_by all_goals exact Termination.decreases
      let rec insertAll (x : Nat) (ys : List Nat) : List (List Nat) :=
        match ys with
        | [] => [[x]]
        | y :: ys =>
            let head := (x :: y :: ys)
            let rest := insertAll x ys
            head :: rest.map (fun tail => y :: tail)
      termination_by 0
      decreasing_by all_goals exact Termination.decreases
      let rec loop (xs : List Nat) : List (List Nat) :=
        match xs with
        | [] => [[]]
        | x :: xs =>
            concatMap (insertAll x) (loop xs)
      termination_by 0
      decreasing_by all_goals exact Termination.decreases
      loop xs

termination_by 0
decreasing_by all_goals exact Termination.decreases
def lastOr (xs : List Nat) (default : Nat) : Nat :=
  match xs.reverse with
  | [] => default
  | x :: _ => x

termination_by 0
decreasing_by all_goals exact Termination.decreases
def solve (n : Nat) : Nat :=
  let rec loopN (n : Nat) : Nat :=
    if n == 0 then
      0
    else
      let digitSum := n * (n + 1) / 2
      if digitSum % 3 == 0 then
        loopN (n - 1)
      else
        let digits := (List.range n).map (fun k => k + 1)
        let rec loopPerm (perms : List (List Nat)) (best : Nat) : Nat :=
          match perms with
          | [] => best
          | p :: ps =>
              let last := lastOr p 0
              let best :=
                if last == 2 || last == 4 || last == 5 || last == 6 || last == 8 then
                  best
                else
                  let val := permToInt p
                  if val > best && isPrime val then val else best
              loopPerm ps best
        termination_by 0
        decreasing_by all_goals exact Termination.decreases
        let best := loopPerm (permutations digits) 0
        if best != 0 then best else loopN (n - 1)
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  loopN n


termination_by 0
decreasing_by all_goals exact Termination.decreases
example : isPrime 2 = true := by
  native_decide

example : isPrime 3 = true := by
  native_decide

example : isPrime 1 = false := by
  native_decide

example : isPrime 9 = false := by
  native_decide
end ProjectEulerSolutions.P41
