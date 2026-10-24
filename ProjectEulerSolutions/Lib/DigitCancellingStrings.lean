import ProjectEulerStatements.Lib.DigitCancellingFractions
import lang.py.FormalSW.Calls
import lang.py.FormalSW.StringReplace
import Mathlib.Tactic.IntervalCases

namespace ProjectEulerSolutions.DigitCancellingStrings
open FormalSW.PyAst
open ProjectEulerStatements.DigitCancellingFractions

set_option Elab.async false
set_option maxRecDepth 10000
set_option maxHeartbeats 1000000

abbrev decimal (n : Nat) : String := toString (n : Int)

/- Each certificate checks one nonzero digit against ninety two-digit strings.
The certificates bridge the actual string interpreter, including first-occurrence
replacement, to the arithmetic predicates used by the independently checked search. -/
abbrev DecimalBridge (n digit : Nat) : Prop :=
  (decimal digit).toList.isInfixOf_internal (decimal n).toList = decide (hasDigit n digit) ∧
  (hasDigit n digit → Strings.replace (decimal n) (decimal digit) "" 1 =
    decimal (remainder n digit))

private theorem decimal_bridge_1 : ∀ n : Fin 90, DecimalBridge (n.val + 10) 1 := by decide +kernel
private theorem decimal_bridge_2 : ∀ n : Fin 90, DecimalBridge (n.val + 10) 2 := by decide +kernel
private theorem decimal_bridge_3 : ∀ n : Fin 90, DecimalBridge (n.val + 10) 3 := by decide +kernel
private theorem decimal_bridge_4 : ∀ n : Fin 90, DecimalBridge (n.val + 10) 4 := by decide +kernel
private theorem decimal_bridge_5 : ∀ n : Fin 90, DecimalBridge (n.val + 10) 5 := by decide +kernel
private theorem decimal_bridge_6 : ∀ n : Fin 90, DecimalBridge (n.val + 10) 6 := by decide +kernel
private theorem decimal_bridge_7 : ∀ n : Fin 90, DecimalBridge (n.val + 10) 7 := by decide +kernel
private theorem decimal_bridge_8 : ∀ n : Fin 90, DecimalBridge (n.val + 10) 8 := by decide +kernel
private theorem decimal_bridge_9 : ∀ n : Fin 90, DecimalBridge (n.val + 10) 9 := by decide +kernel

theorem decimal_bridge (n digit : Nat) (lower : 10 ≤ n) (upper : n < 100)
    (positive : 0 < digit) (single : digit < 10) : DecimalBridge n digit := by
  let index : Fin 90 := ⟨n - 10, by omega⟩
  have restore : index.val + 10 = n := by dsimp [index]; omega
  interval_cases digit <;>
    first
    | simpa only [restore] using decimal_bridge_1 index
    | simpa only [restore] using decimal_bridge_2 index
    | simpa only [restore] using decimal_bridge_3 index
    | simpa only [restore] using decimal_bridge_4 index
    | simpa only [restore] using decimal_bridge_5 index
    | simpa only [restore] using decimal_bridge_6 index
    | simpa only [restore] using decimal_bridge_7 index
    | simpa only [restore] using decimal_bridge_8 index
    | simpa only [restore] using decimal_bridge_9 index

theorem one_digit_facts : ∀ n : Fin 10,
    (decimal n.val).length = 1 ∧
    Calls.integerConversion [.string (decimal n.val)] = some (.ok (.int n.val)) ∧
    (decimal n.val = "0" ↔ n.val = 0) := by decide +kernel


end ProjectEulerSolutions.DigitCancellingStrings
