import «.formalsw».solvers.«22.py.ast»
import lang.py.FormalSW.HeapRuntimeCostSumGenerators

/-! Runtime of the optimized source's unguarded ordinal sum. All Unicode
strings are permitted, so the result is an integer and may be negative. -/
namespace solvers.«22».runtimeCost.Name
open FormalSW.PyAst HeapCalls.Runtime

abbrev program := Solvers.P22Ast.module
def number (value : Int) : Mutable.Value := .scalar (.int value)
def character (c : Char) : Mutable.Value := .scalar (.string (String.ofList [c]))

def definition : List Arg × List Stmt :=
  (findFunctionIn program.body "name_value").getD ([], [])

def expression : Expr :=
  match ((executableBody definition.2)[0]? : Option Stmt) with
  | some (.return value) => value
  | _ => .constant .none

def element : Expr :=
  match expression with
  | .sumGenerator value _ _ => value
  | _ => .constant .none

theorem body_eq : executableBody definition.2 = [.return expression] := rfl

theorem expression_eq : expression =
    .sumGenerator element [("c", .name "name", [])] (.constant (.int 0)) := rfl

theorem element_eq : element =
    .binOp (.binOp (.builtinCall .ord [.name "c"]) .sub
      (.builtinCall .ord [.constant (.string "A")])) .add (.constant (.int 1)) := rfl

def letterValue (c : Char) : Int := (c.toNat : Int) - 65 + 1

def value (text : String) : Int :=
  text.toList.foldl (fun total c => total + letterValue c) 0

private theorem ordinal_cost (c : Char) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (argument : Expr) (cost : Nat)
    (evaluated : EvaluatesIn .word program env argument heap cost (character c) heap) :
    EvaluatesIn .word program env (.builtinCall .ord [argument]) heap (cost + 4)
      (number c.toNat) heap := by
  apply EvaluatesIn.builtin_single (builtin := .ord) (work := 1)
    (by intro h; cases h) (by intro h; cases h) evaluated
  · change HeapCalls.scalarResult (some (Strings.ord (.string (String.ofList [c])))) heap = _
    rw [Strings.ord_character]
    rfl
  · simp [operation, scalarValues, Calls.Runtime.primitive, character]

/-- Both `ord` calls occur in each actual generator element. Subtraction is
signed, with no uppercase precondition or natural-number truncation. -/
theorem element_cost (c : Char) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (input : env "c" = some (character c)) :
    EvaluatesIn .word program env element heap 19 (number (letterValue c)) heap := by
  have ordinal : EvaluatesIn .word program env (.builtinCall .ord [.name "c"])
      heap 5 (number c.toNat) heap :=
    ordinal_cost c heap env (.name "c") 1 (EvaluatesIn.lookup input)
  have lower : EvaluatesIn .word program env (.builtinCall .ord [.constant (.string "A")])
      heap 5 (number 65) heap :=
    ordinal_cost 'A' heap env (.constant (.string "A")) 1 EvaluatesIn.constant
  have difference : EvaluatesIn .word program env
      (.binOp (.builtinCall .ord [.name "c"]) .sub (.builtinCall .ord [.constant (.string "A")]))
      heap 14 (number ((c.toNat : Int) - 65)) heap := by
    apply EvaluatesIn.binary (work := 1) ordinal lower
    · simp [Mutable.binary, scalarBinary, number]
    · rfl
  rw [element_eq]
  apply EvaluatesIn.binary (work := 1) difference EvaluatesIn.constant
  · simp [Mutable.binary, scalarBinary, number, letterValue]
  · rfl

/-- Lazy traversal charges one constant-cost element computation and the
generator's summation work per Unicode character, preserving any initial heap. -/
theorem function_run (text : String) (heap : Mutable.Heap) :
    ReturnsIn .word program "name_value" [.scalar (.string text)] heap
      (31 * text.length + 21) (number (value text)) heap := by
  let entry := HeapCalls.functionEnv definition.1 definition.2 [.scalar (.string text)] []
  have sequence : Mutable.sequence (.scalar (.string text)) heap =
      some (.ok (text.toList.map character), heap) := by
    simp [Mutable.sequence, character]
  have priced : ConstantSequenceRead program heap (.scalar (.string text)) := by
    intro name position body otherwise following env stack
    rfl
  have evaluated : EvaluatesIn .word program entry expression heap
      (31 * text.length + 17) (number (value text)) heap := by
    rw [expression_eq]
    apply EvaluatesIn.mono (small := 1 + 1 + (31 * text.toList.length + 7) + 8)
    · apply EvaluatesIn.sum_generator (size := text.length)
        (ordinary := by intro expressions impossible; cases impossible)
        (iterated := EvaluatesIn.lookup (name := "name") rfl)
        (sequence := sequence) (length := rfl) (started := EvaluatesIn.constant)
      have loop := sum_sequence_loop (module := program) (source := .scalar (.string text))
        (values := text.toList) (encode := character) (number := letterValue)
        (name := "c") (element := element)
        (env := HeapCalls.setLocal (HeapCalls.comprehensionEnv entry [("c", .name "name", [])])
          HeapCalls.sumAccumulator (number 0)) (initial := 0)
        text.toList 0 19 (by decide) sequence priced
        (fun env c lookup => element_cost c heap env lookup) rfl
        (by simp [HeapCalls.setLocal, number])
      simpa only [number, value] using loop
    · simp only [String.length_toList]
      omega
  have returned : BlockReturnsIn .word program entry (executableBody definition.2)
      heap (31 * text.length + 18) (number (value text)) heap := by
    rw [body_eq]
    exact BlockReturnsIn.return_value evaluated
  have called : ReturnsIn .word program "name_value" [.scalar (.string text)] heap
      (3 + (31 * text.length + 18)) (number (value text)) heap :=
    BlockReturnsIn.function (args := definition.1) (body := definition.2)
      (values := [.scalar (.string text)]) (completed := [.scalar (.string text)]) rfl rfl returned
  exact called.mono (by omega)

theorem function_exists (text : String) (heap : Mutable.Heap) :
    ∃ answer : Int, ReturnsIn .word program "name_value" [.scalar (.string text)] heap
      (31 * text.length + 21) (number answer) heap :=
  ⟨value text, function_run text heap⟩

end solvers.«22».runtimeCost.Name
