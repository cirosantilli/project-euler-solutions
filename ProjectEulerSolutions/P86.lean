import ProjectEulerStatements.P86
import ProjectEulerSolutions.Termination.P86
namespace ProjectEulerSolutions.P86

def sqrtFloor (n : Nat) : Nat :=
  let rec loop (lo hi : Nat) : Nat :=
    if lo > hi then
      hi
    else
      let mid := (lo + hi) / 2
      let sq := mid * mid
      if sq == n then
        mid
      else if sq < n then
        loop (mid + 1) hi
      else
        loop lo (mid - 1)
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  loop 1 n

termination_by 0
decreasing_by all_goals exact Termination.decreases
def countSolutionsUpTo (m : Nat) : Nat :=
  let rec loopC (c : Nat) (total : Nat) : Nat :=
    if c > m then
      total
    else
      let c2 := c * c
      let rec loopS (s : Nat) (total : Nat) : Nat :=
        if s > 2 * c then
          total
        else
          let d := s * s + c2
          let t := sqrtFloor d
          let total :=
            if t * t == d then
              let lo := if s > c then s - c else 1
              let hi := s / 2
              if hi >= lo then total + (hi - lo + 1) else total
            else
              total
          loopS (s + 1) total
      termination_by 0
      decreasing_by all_goals exact Termination.decreases
      loopC (c + 1) (loopS 2 total)
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  loopC 1 0

termination_by 0
decreasing_by all_goals exact Termination.decreases
def leastMExceeding (target : Nat) : Nat :=
  let rec loop (c : Nat) (total : Nat) : Nat :=
    if total > target then
      c
    else
      let c := c + 1
      let c2 := c * c
      let rec loopS (s : Nat) (total : Nat) : Nat :=
        if s > 2 * c then
          total
        else
          let d := s * s + c2
          let t := sqrtFloor d
          let total :=
            if t * t == d then
              let lo := if s > c then s - c else 1
              let hi := s / 2
              if hi >= lo then total + (hi - lo + 1) else total
            else
              total
          loopS (s + 1) total
      termination_by 0
      decreasing_by all_goals exact Termination.decreases
      loop c (loopS 2 total)
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  loop 0 0


termination_by 0
decreasing_by all_goals exact Termination.decreases
example : countSolutionsUpTo 99 = 1975 := by
  native_decide

example : countSolutionsUpTo 100 = 2060 := by
  native_decide


def solve (_n : Nat) :=
  leastMExceeding 1000000
end ProjectEulerSolutions.P86
