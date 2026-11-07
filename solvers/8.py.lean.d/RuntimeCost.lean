import solvers.«8.py.lean.d».OuterRuntime

namespace solvers.«8».runtimeCost
open FormalSW.PyAst HeapCalls.Runtime
private abbrev program := Solvers.P8Ast.module
set_option maxRecDepth 4096
set_option maxHeartbeats 1000000

theorem source_run (digits : List Int) (width : Nat) (heap : Mutable.Heap) (address : Nat)
    (found : heap[address]? = some (.list (ints digits))) :
    ∃ answer : Int, ReturnsIn .word program "solve" [.ref address, .scalar (.int width)] heap
      (300*digits.length+90) (.scalar (.int answer)) heap := by
  let args : List Arg := [⟨"digits", some (.subscript (.name "list") (.name "int")), none⟩,
    ⟨"k", some (.name "int"), none⟩]
  let env := HeapCalls.argumentEnv args [.ref address, .scalar (.int width)]
  let counted := HeapCalls.setLocal env "digit_count" (.scalar (.int digits.length))
  have lengthRun : EvaluatesIn .word program env (.builtinCall .len [.name "digits"]) heap 5
      (.scalar (.int digits.length)) heap := by
    apply EvaluatesIn.builtin_single (work := 1) (by intro impossible; cases impossible)
      (by intro impossible; cases impossible) (EvaluatesIn.lookup rfl)
    · simp [HeapCalls.primitive, Mutable.length, found, ints]
    · simp [operation, sequenceLength, found, ints]
  have guard : EvaluatesIn .word program counted (.compare (.name "k") [(.eq, .constant (.int 0))]) heap 7
      (.scalar (.bool (decide (width = 0)))) heap :=
    EvaluatesIn.compare (op := .eq) (env := counted) (av := .scalar (.int width)) (bv := .scalar (.int 0))
      (value := .scalar (.bool (decide (width = 0)))) (middle := heap) (afterRight := heap)
      (EvaluatesIn.lookup rfl) EvaluatesIn.constant
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]) rfl
  by_cases zero : width = 0
  · refine ⟨if digits.isEmpty then 0 else 1, ?_⟩
    apply ReturnsIn.mono (small := 32) _ (by omega)
    apply BlockReturnsIn.function (args := args) (values := [.ref address, .scalar (.int width)]) (budget := 27) rfl rfl
    apply BlockReturnsIn.assign (ca := 5) (cb := 20) lengthRun
    apply BlockReturnsIn.branch (cc := 7) (cb := 11) (value := .bool true) (flag := true)
    · simpa [zero] using guard
    · rfl
    · apply BlockReturnsIn.return_value (cost := 10)
      apply EvaluatesIn.conditional (ct := 7) (cb := 1) (value := .bool digits.isEmpty) (flag := digits.isEmpty)
      · apply EvaluatesIn.compare (work := 2) (av := .scalar (.int digits.length)) (bv := .scalar (.int 0))
          (EvaluatesIn.lookup rfl) EvaluatesIn.constant
        · cases digits <;> simp [HeapCalls.compare, scalarCompare, evalCmpOp]
          omega
        · rfl
      · rfl
      · cases digits.isEmpty <;> exact EvaluatesIn.constant
  · let ready := HeapCalls.setLocal (HeapCalls.setLocal counted "best" (.scalar (.int 0))) "i" (.scalar (.int 0))
    obtain ⟨answer, loop⟩ := outerRuntime.outer digits heap address width 0 ready 0 found rfl rfl rfl rfl rfl (by omega)
    refine ⟨answer, ?_⟩
    apply ReturnsIn.mono (small := 5+(5+(7+(1+(1+(300*(digits.length-0)+60)+2)+2)+2)+2)) _ (by omega)
    apply BlockReturnsIn.function (args := args) rfl rfl
    apply BlockReturnsIn.assign (ca := 5) lengthRun
    apply BlockReturnsIn.branch (cc := 7) (value := .bool false) (flag := false)
    · simpa [zero] using guard
    · rfl
    · apply BlockReturnsIn.assign (ca := 1) EvaluatesIn.constant
      apply BlockReturnsIn.assign (ca := 1) EvaluatesIn.constant
      exact loop

theorem source_bound (digits : List Int) (width : Nat) :
    TimeBound .word program "solve" [.ref 0, .scalar (.int width)] [.list (ints digits)]
      (300*digits.length+100) := by
  obtain ⟨answer, called⟩ := source_run digits width [.list (ints digits)] 0 rfl
  obtain ⟨cost, trace, paid⟩ := called 1 []
  have terminal : RuntimeCost.Finishes (HeapCalls.step program 1) (charge .word program)
      ⟨.value (.scalar (.int answer)), [], [.list (ints digits)]⟩ 1
      (.ok ⟨.ok (.scalar (.int answer)), [], [.list (ints digits)], 1⟩) := .done rfl rfl
  refine ⟨cost+1, _, trace.finishes terminal, ?_⟩
  simp only [potential] at paid
  omega

end solvers.«8».runtimeCost

namespace solvers.«8»
open FormalSW.PyAst

theorem solve.«BigO(n, n = len(digits))» :
    FormalSW.PyAst.HeapCalls.Runtime.IsBigO .word Solvers.P8Ast.module "solve"
      (fun input : List Int × Int => [.ref 0, .scalar (.int input.2)]) (fun input : List Int × Int => [.list (input.1.map (fun value => .scalar (.int value)))])
      (fun (input : List Int × Int) => (FormalSW.PyAst.HeapCalls.Condition Solvers.P8Ast.module (fun _arg => if _arg = "k" then some (.scalar (.int input.2)) else none) Solvers.P8Ast.requires_solve_0 [.list (input.1.map (fun value => .scalar (.int value)))])) (fun input : List Int × Int => input.1.length) (fun n : Nat => n)
    := by
  refine ⟨400, 1, ?_⟩
  rintro ⟨digits, width⟩ valid large
  have checked : HeapCalls.Evaluates Solvers.P8Ast.module
      (fun name => if name = "k" then some (.scalar (.int width)) else none)
      Solvers.P8Ast.requires_solve_0 [.list (ints digits)]
      (.scalar (.bool (decide (0 ≤ width)))) [.list (ints digits)] := by
    apply HeapCalls.Evaluates.compare (av := .scalar (.int width)) (bv := .scalar (.int 0))
      (HeapCalls.Evaluates.lookup rfl) HeapCalls.Evaluates.constant
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  have nonnegative : 0 ≤ width := by simpa using checked.condition_iff.mp valid
  have bounded := runtimeCost.source_bound digits width.toNat
  rw [Int.toNat_of_nonneg nonnegative] at bounded
  apply bounded.mono
  change 1 ≤ digits.length at large
  change 300*digits.length+100 ≤ 400*digits.length
  omega

end solvers.«8»
