import ProjectEulerStatements.P14
import ProjectEulerSolutions.Termination.P14
namespace ProjectEulerSolutions.P14

def collatzNext (n : Nat) : Nat :=
  if n % 2 == 1 then 3 * n + 1 else n / 2

def collatzLength (n : Nat) : Nat :=
  if n == 1 then
    1
  else
    1 + collatzLength (collatzNext n)
termination_by 0
decreasing_by all_goals exact Termination.decreases

def collatzLengthMemo (n limit : Nat) (cache : Array Nat) : Nat × Array Nat :=
  let rec loop (m : Nat) (path : List Nat) (cache : Array Nat) : Nat × Array Nat × List Nat :=
    if m <= limit then
      let cm := cache[m]!
      if cm != 0 then
        (cm, cache, path)
      else
        loop (collatzNext m) (m :: path) cache
    else
      loop (collatzNext m) (m :: path) cache
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  let (len, cache1, path) := loop n [] cache
  let rec propagate (path : List Nat) (len : Nat) (cache : Array Nat) : Nat × Array Nat :=
    match path with
    | [] => (len, cache)
    | v :: vs =>
        let len' := len + 1
        let cache' := if v <= limit then cache.set! v len' else cache
        propagate vs len' cache'
  propagate path len cache1

def longestCollatzUnder (limit : Nat) : Nat × Nat :=
  let cache0 := (Array.replicate (limit + 1) 0).set! 1 1
  let rec loop (n bestN bestLen : Nat) (cache : Array Nat) : Nat × Nat :=
    if n >= limit then
      (bestN, bestLen)
    else
      let (len, cache') := collatzLengthMemo n limit cache
      let (bestN', bestLen') := if len > bestLen then (n, len) else (bestN, bestLen)
      loop (n + 1) bestN' bestLen' cache'
  termination_by limit - n
  decreasing_by
    simp_wf
    exact Termination.sub_succ_lt_sub (by assumption)
  loop 1 1 1 cache0


def solve (limit : Nat) : Nat :=
  (longestCollatzUnder limit).1

example : collatzLength 1 = 1 := by
  native_decide

example : collatzLength 13 = 10 := by
  native_decide
end ProjectEulerSolutions.P14
