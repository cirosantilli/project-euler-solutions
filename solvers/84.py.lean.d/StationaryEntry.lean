import solvers.«84.py.lean.d».TransitionAccess
import ProjectEulerSolutions.Lib.MonopolySystem

namespace solvers.«84».stationary_weights
open FormalSW.PyAst HeapCalls ProjectEulerSolutions NaturalDictionary
set_option Elab.async false

def definition := (findFunctionIn program.body "stationary_weights").getD ([], [])

def diagonalExpression : Expr := .ifExp (.compare (.name "source") [(.eq, .name "target")])
  (.name "denominator") (.constant (.int 0))

def coefficientExpression : Expr :=
  .binOp (.call (.attribute (.subscript (.name "transitions") (.name "source")) "get")
    [.name "target", .constant (.int 0)]) .sub diagonalExpression

def rowExpression : Expr :=
  .binOp (.listComp coefficientExpression [("source", .call (.name "range") [.name "size"], [])])
    .add (.list [.constant (.int 0)])

def matrixExpression : Expr :=
  .listComp rowExpression [("target", .call (.name "range") [.binOp (.name "size") .sub (.constant (.int 1))], [])]

theorem setup_source : (executableBody definition.2).take 2 =
    [.assign [.name "size"] (.builtinCall .len [.name "transitions"]), .assign [.name "matrix"] matrixExpression] := rfl

theorem size_result (sides : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (value : Mutable.Value) (view : transitions.View sides heap value) (getTransitions : env "transitions" = some value) :
    Evaluates program env (.builtinCall .len [.name "transitions"]) heap (number 120) heap globals := by
  obtain ⟨root, items, _, rfl, found, length, _, _⟩ := view.at 0 (by decide)
  apply Evaluates.builtin (.cons (Evaluates.lookup getTransitions) .nil)
  simp [HeapCalls.primitive, Mutable.length, found, length, number]

theorem diagonal_result (denominator : Int) (source target : Fin 120)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getSource : env "source" = some (number source.val)) (getTarget : env "target" = some (number target.val))
    (getDenominator : env "denominator" = some (.scalar (.int denominator))) :
    Evaluates program env diagonalExpression heap (.scalar (.int (if source = target then denominator else 0))) heap globals := by
  have tested : Evaluates program env (.compare (.name "source") [(.eq, .name "target")]) heap
      (.scalar (.bool (decide (source = target)))) heap globals := by
    apply Evaluates.compare (Evaluates.lookup getSource) (Evaluates.lookup getTarget)
    simp [HeapCalls.compare, scalarCompare, evalCmpOp, Fin.ext_iff]
  apply Evaluates.conditional (flag := decide (source = target)) tested rfl
  by_cases same : source = target
  · simpa [same] using Evaluates.lookup (module := program) (heap := heap) (globals := globals) getDenominator
  · simpa [same] using (Evaluates.constant (module := program) (env := env) (heap := heap) (globals := globals)
      (value := .int 0))

/-- Each entry of the source comprehension is the corresponding integer stationary coefficient. -/
theorem coefficient_result (sides : Nat) (source target : Fin 120) (ordinary : target ≠ 119)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (value : Mutable.Value) (view : transitions.View sides heap value)
    (getTransitions : env "transitions" = some value) (getSource : env "source" = some (number source.val))
    (getTarget : env "target" = some (number target.val))
    (getDenominator : env "denominator" = some (.scalar (.int (256 * (sides : Int) * sides)))) :
    Evaluates program env coefficientExpression heap (.scalar (.int (Monopoly.coefficients sides target source))) heap globals := by
  rw [Monopoly.coefficients, if_neg ordinary]
  apply Evaluates.binary (transitions.get_result sides source target heap env globals value view getTransitions getSource getTarget)
    (diagonal_result _ source target heap env globals getSource getTarget getDenominator)
  rfl

end solvers.«84».stationary_weights
