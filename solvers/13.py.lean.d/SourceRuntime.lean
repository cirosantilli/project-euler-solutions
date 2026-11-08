import solvers.«13.py.lean.d».LeadingRuntime
import lang.py.FormalSW.HeapRuntimeCostSetup

namespace solvers.«13».runtimeCost
open FormalSW.PyAst HeapCalls HeapCalls.Runtime
open HeapCalls.Runtime.NestedIntList (encode)
private abbrev program := Solvers.P13Ast.module

/-- The original adapter formats its inputs and invokes the string-based helper. -/
theorem source_run (items : List Int) (k : Nat) (heap : Heap) (address : Nat)
    (found : heap[address]? = some (.list (encode items)))
    (nonnegative : 0 ≤ items.foldl Int.add 0) :
    ∃ answer : Int, ReturnsIn .word program "solve" [.ref address, .scalar (.int k)] heap
      (40*IntegerListWork.size items+100) (.scalar (.int answer)) (heap ++ [.list (texts items)]) := by
  let args := ((findFunctionIn program.body "solve").getD ([], [])).1
  let body := ((findFunctionIn program.body "solve").getD ([], [])).2
  let env := functionEnv args body [.ref address, .scalar (.int k)] []
  let middle := heap ++ [.list (texts items)]
  let leading := String.ofList ((toString (items.foldl Int.add 0)).toList.take k)
  let after := setLocal env "leading" (.scalar (.string leading))
  have formatted := formatting_cost heap address items env found rfl
  have helper := first_digits_cost middle heap.length items k (by simp [middle]) nonnegative
  have computed := EvaluatesIn.call
    (ArgumentsIn.cons formatted (ArgumentsIn.cons (EvaluatesIn.lookup (env := env) (name := "k") rfl) ArgumentsIn.nil)) helper
  have length : leading.length ≤ IntegerListWork.size items+1 := by
    have bound := IntegerListWork.sum_decimal_length items
    rw [Int.natAbs_of_nonneg nonnegative] at bound
    simp only [leading, String.length_ofList, List.length_take, String.length_toList]
    omega
  obtain ⟨answer, returned⟩ := leading_cost (items.foldl Int.add 0) k
    (IntegerListWork.size items) middle after nonnegative rfl length
  refine ⟨answer, ?_⟩
  have called := BlockReturnsIn.function (module := program) (name := "solve")
    (args := args) (body := body) (values := [.ref address, .scalar (.int k)]) rfl rfl
    (BlockReturnsIn.assign computed (BlockReturnsIn.return_value (following := []) returned))
  intro anchors stack
  exact (called anchors stack).mono (by
    have arity : args.length = 2 := rfl
    simp only [List.length_cons, List.length_nil, arity]
    omega)

theorem source_bound (items : List Int) (k : Nat) (nonnegative : 0 ≤ items.foldl Int.add 0) :
    TimeBound .word program "solve" [.ref 0, .scalar (.int k)] [.list (encode items)]
      (40*IntegerListWork.size items+101) := by
  obtain ⟨answer, run⟩ := source_run items k [.list (encode items)] 0 rfl nonnegative
  exact run.timeBound_append

end solvers.«13».runtimeCost
