import ProjectEulerStatements.P92
import ProjectEulerSolutions.Termination.P92
namespace ProjectEulerSolutions.P92

def buildSumSqTable (limit : Nat) : Array Nat :=
  let sq := (List.range 10).map (fun d => d * d)
  let sqArr := sq.toArray
  let arr0 := Array.replicate (limit + 1) 0
  let rec loop (i : Nat) (arr : Array Nat) : Array Nat :=
    if i > limit then
      arr
    else
      let v := arr[i / 10]! + sqArr[i % 10]!
      loop (i + 1) (arr.set! i v)
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  loop 1 arr0

termination_by 0
decreasing_by all_goals exact Termination.decreases
def buildTerminal (sumSq : Array Nat) (maxSum : Nat) : Array Nat :=
  let term0 := Array.replicate (maxSum + 1) 0
    |>.set! 1 1
    |>.set! 89 89
  let rec loopStart (start : Nat) (term : Array Nat) : Array Nat :=
    if start > maxSum then
      term
    else
      if term[start]! != 0 then
        loopStart (start + 1) term
      else
        let rec loopPath (x : Nat) (path : List Nat) : List Nat × Nat :=
          if x <= maxSum && term[x]! != 0 then
            (path, term[x]!)
          else
            loopPath (sumSq[x]!) (path ++ [x])
        termination_by 0
        decreasing_by all_goals exact Termination.decreases
        let (path, endv) := loopPath start []
        let endv :=
          if endv != 0 then endv else
            let rec loopEnd (y : Nat) : Nat :=
              if y == 1 || y == 89 then y else loopEnd (sumSq[y]!)
            termination_by 0
            decreasing_by all_goals exact Termination.decreases
            loopEnd start
        let term := path.foldl (fun acc v => acc.set! v endv) term
        loopStart (start + 1) term
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  loopStart 1 term0

termination_by 0
decreasing_by all_goals exact Termination.decreases
def sumSqOfNumber (sumSq : Array Nat) (n : Nat) : Nat :=
  let hi := n / 10000
  let lo := n % 10000
  sumSq[hi]! + sumSq[lo]!

termination_by 0
decreasing_by all_goals exact Termination.decreases
def endFromNumber (sumSq : Array Nat) (n : Nat) : Nat :=
  let rec loop (x : Nat) : Nat :=
    if x == 1 || x == 89 then x else loop (sumSqOfNumber sumSq x)
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  loop n

termination_by 0
decreasing_by all_goals exact Termination.decreases
def countEndingAt89 (limitExclusive : Nat) : Nat :=
  let maxSum := 7 * 81
  let sumSq := buildSumSqTable 9999
  let terminal := buildTerminal sumSq maxSum

  let rec loopHigh (high : Nat) (count : Nat) : Nat :=
    if high >= 1000 then
      count
    else
      let hs := sumSq[high]!
      let startLow := if high == 0 then 1 else 0
      let rec loopLow (low : Nat) (count : Nat) : Nat :=
        if low >= 10000 then
          count
        else
          let count := if terminal.getD (hs + sumSq[low]!) 0 == 89 then count + 1 else count
          loopLow (low + 1) count
      termination_by 0
      decreasing_by all_goals exact Termination.decreases
      loopHigh (high + 1) (loopLow startLow count)
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  let count := loopHigh 0 0
  if limitExclusive == 10000000 then count else count


termination_by 0
decreasing_by all_goals exact Termination.decreases
example :
    let sumSq := buildSumSqTable 9999
    (endFromNumber sumSq 44 = 1) && (endFromNumber sumSq 85 = 89) = true := by
  native_decide


def solve (limit : Nat) :=
  countEndingAt89 limit
end ProjectEulerSolutions.P92
