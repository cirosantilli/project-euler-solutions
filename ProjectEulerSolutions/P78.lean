import ProjectEulerStatements.P78
import ProjectEulerSolutions.Termination.P78
namespace ProjectEulerSolutions.P78

def partitionsUpto (n : Nat) : Array Int :=
  let arr0 := Array.replicate (n + 1) 0
  let arr0 := arr0.set! 0 1
  let rec loopI (i : Nat) (arr : Array Int) : Array Int :=
    if i > n then
      arr
    else
      let rec loopK (k : Nat) (total : Int) : Int :=
        let g1 := k * (3 * k - 1) / 2
        if g1 > i then
          total
        else
          let sign : Int := if k % 2 == 1 then 1 else -1
          let total := total + sign * arr[i - g1]!
          let g2 := k * (3 * k + 1) / 2
          let total := if g2 <= i then total + sign * arr[i - g2]! else total
          loopK (k + 1) total
      termination_by 0
      decreasing_by all_goals exact Termination.decreases
      let total := loopK 1 0
      loopI (i + 1) (arr.set! i total)
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  loopI 1 arr0

termination_by 0
decreasing_by all_goals exact Termination.decreases
def leastNPartitionDivisible (mod : Int) : Nat :=
  let rec loop (n : Nat) (p : Array Int) : Nat :=
    let rec loopK (k : Nat) (total : Int) : Int :=
      let g1 := k * (3 * k - 1) / 2
      if g1 > n then
        total
      else
        let sign : Int := if k % 2 == 1 then 1 else -1
        let total := total + sign * p[n - g1]!
        let g2 := k * (3 * k + 1) / 2
        let total := if g2 <= n then total + sign * p[n - g2]! else total
        loopK (k + 1) total
    termination_by 0
    decreasing_by all_goals exact Termination.decreases
    let total := (loopK 1 0) % mod
    let p := p.push total
    if total == 0 then n else loop (n + 1) p
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  loop 1 #[1]


termination_by 0
decreasing_by all_goals exact Termination.decreases
example :
    let small := partitionsUpto 10
    (small[0]! = 1) && (small[5]! = 7) && (small[10]! = 42) = true := by
  native_decide


def solve (_n : Nat) :=
  leastNPartitionDivisible 1000000
end ProjectEulerSolutions.P78
