import solvers.«12.py.lean.d».Source
import lang.py.FormalSW.HeapRuntimeCostClasses

namespace solvers.«12».runtimeCost
open FormalSW.PyAst HeapCalls HeapCalls.Runtime

/-- Allocate the source object and its one-element prime list, then execute the
annotated field assignment and implicit None return. -/
theorem constructor_cost (heap : Heap) :
    ReturnsIn .word program "PrimeTable" [] heap 21 (.ref heap.length) (initialized heap) := by
  apply ReturnsIn.constructor (origin := origin) (args := initMethod.1) (body := initMethod.2)
    (budget := 16) rfl rfl rfl rfl rfl
  let env := functionEnv initMethod.1 initMethod.2 [.ref heap.length] []
  let before := heap ++ [.instance origin "PrimeTable" []]
  let withList := before ++ [.list (primeValues [2])]
  have listed : EvaluatesIn .word program env (.list [.constant (.int 2)]) before 7
      (.ref (heap.length+1)) withList := by
    simpa [before, withList, primeValues] using
      (EvaluatesIn.list (arithmetic := .word) (module := program) (env := env) (heap := before)
        (ArgumentsIn.cons (EvaluatesIn.constant (value := .int 2)) ArgumentsIn.nil))
  have found : withList[heap.length]? = some (.instance origin "PrimeTable" []) := by simp [withList, before]
  have written : Mutable.writeField (.ref heap.length) "primes" (.ref (heap.length+1)) withList =
      some (.ok (.scalar .none), initialized heap) :=
    Mutable.writeField_result found (by decide)
  have credits : heapPotential (initialized heap) = heapPotential withList := by
    have equality := heapPotential_set withList heap.length (.instance origin "PrimeTable" [])
      (.instance origin "PrimeTable" [("primes", .ref (heap.length+1))]) found
    simpa only [objectPotential, Nat.add_zero] using equality
  have assigned : BlockContinuesIn .word program env (executableBody initMethod.2) before 15 env (initialized heap) := by
    apply BlockContinuesIn.annAttribute (cr := 7) (ct := 1) (work := 2) listed (EvaluatesIn.lookup rfl) written
    · simp [RecordWork.write, found, RecordWork.lookup]
    · exact credits
  exact assigned.returns BlockReturnsIn.fallthrough

end solvers.«12».runtimeCost
