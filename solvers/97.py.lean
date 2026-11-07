import «.formalsw».solvers.«97.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p97.py»

namespace solvers.«97»
open FormalSW.PyAst FormalSW.PyAst.Calls
open ProjectEulerSolutions.ModularPower
open data.«project-euler-statements».data.py.project_euler_statements.p97 (answer)
set_option Elab.async false
private abbrev program := Solvers.P97Ast.module

/-- Binary modular exponentiation and the reference multiplication loop return the same integer. -/
theorem solve.result : ReturnsValue program "solve" [] (.int answer) := by
  apply BlockReturns.function rfl rfl
  apply BlockReturns.annAssign (value := .int (10^10))
  · exact Evaluates.binary Evaluates.constant Evaluates.constant rfl
  apply BlockReturns.return_value
  apply Evaluates.binary (av := .int (28433 * power 2 7830457 (10^10) + 1)) (bv := .int (10^10))
  · apply Evaluates.binary (av := .int (28433 * power 2 7830457 (10^10))) (bv := .int 1)
    · apply Evaluates.binary (av := .int 28433) (bv := .int (power 2 7830457 (10^10)))
      · exact Evaluates.constant
      · apply Evaluates.builtin (.cons Evaluates.constant (.cons Evaluates.constant (.cons (Evaluates.lookup rfl) .nil)))
        rw [show power 2 7830457 (10^10) = (2 : Int)^7830457 % (10^10) from
          power_positive 2 7830457 (by decide) (10^10) (by decide)]
        have reduced : FormalSW.Py.Builtin.Int.powMod 2 7830457 (10^10) =
            .ok ((2 : Int)^7830457 % (10^10)) :=
          FormalSW.Py.Builtin.Int.powMod_nonnegative 2 7830457 (10^10) (by decide)
        simp only [Primitive.apply, modularPower, scalarNumber, reduced]
      · rfl
    · exact Evaluates.constant
    · rfl
  · exact Evaluates.lookup rfl
  · rfl

theorem solve.«Equiv(naive)» :
    (Calls.function program "solve" []).Equiv (Calls.function P97Ast.module "naive" []) :=
  solve.result.equiv data.«project-euler-statements».data.py.project_euler_statements.p97.naive.result

theorem solve.«Supported» :
    (Calls.function program "solve" []).Supported ∧ (Calls.function P97Ast.module "naive" []).Supported :=
  ⟨solve.result.returns.supported,
    data.«project-euler-statements».data.py.project_euler_statements.p97.naive.«Supported»⟩

end solvers.«97»
