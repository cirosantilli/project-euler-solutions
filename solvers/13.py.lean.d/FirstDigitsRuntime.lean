import solvers.«13.py.lean.d».ParsingRuntime
import lang.py.FormalSW.HeapRuntimeCostStrings

namespace solvers.«13».runtimeCost
open FormalSW.PyAst HeapCalls HeapCalls.Runtime
private abbrev program := Solvers.P13Ast.module

/-- The helper parses each input string, adds the values, formats the sum and copies its prefix. -/
theorem first_digits_cost (heap : Heap) (address : Nat) (items : List Int) (k : Nat)
    (found : heap[address]? = some (.list (texts items)))
    (nonnegative : 0 ≤ items.foldl Int.add 0) :
    ReturnsIn .word program "first_digits" [.ref address, .scalar (.int k)] heap
      (21*IntegerListWork.size items+60)
      (.scalar (.string (String.ofList ((toString (items.foldl Int.add 0)).toList.take k)))) heap := by
  let args := ((findFunctionIn program.body "first_digits").getD ([], [])).1
  let body := ((findFunctionIn program.body "first_digits").getD ([], [])).2
  let env := functionEnv args body [.ref address, .scalar (.int k)] []
  let total := items.foldl Int.add 0
  let after := setLocal env "total" (.scalar (.int total))
  have sum := summation_cost heap address items env found rfl
  have formatted : EvaluatesIn .word program after (.builtinCall .str [.name "total"]) heap
      (IntegerListWork.size items+6) (.scalar (.string (toString total))) heap := by
    have run := EvaluatesIn.builtin_single (arithmetic := .word) (module := program)
      (builtin := .str) (heap := heap) (work := PrimitiveWork.bitLength total+1)
      (by intro h; cases h) (by intro h; cases h)
      (EvaluatesIn.lookup (env := after) (name := "total") rfl) (by rfl) rfl
    exact run.mono (by have width := IntegerListWork.sum_width items; dsimp [total] at *; omega)
  have sliced := EvaluatesIn.string_prefix_word formatted
    (EvaluatesIn.lookup (env := after) (name := "k") (by rfl))
  have length : (toString total).length ≤ IntegerListWork.size items+1 := by
    have bound := IntegerListWork.sum_decimal_length items
    simpa only [Int.natAbs_of_nonneg nonnegative] using bound
  have returned : BlockReturnsIn .word program env (executableBody body) heap
      (21*IntegerListWork.size items+55)
      (.scalar (.string (String.ofList ((toString total).toList.take k)))) heap := by
    have composed := BlockReturnsIn.assign sum (BlockReturnsIn.return_value (following := []) sliced)
    exact composed.mono (by omega)
  have called := BlockReturnsIn.function (module := program) (name := "first_digits")
    (args := args) (body := body) (values := [.ref address, .scalar (.int k)]) rfl rfl returned
  intro anchors stack
  exact (called anchors stack).mono (by
    change 2+2+1+(21*IntegerListWork.size items+55) ≤ 21*IntegerListWork.size items+60
    omega)

end solvers.«13».runtimeCost
