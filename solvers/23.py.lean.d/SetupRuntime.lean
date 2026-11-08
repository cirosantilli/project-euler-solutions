import solvers.«23.py.lean.d».ScanRuntime
import lang.py.FormalSW.HeapRuntimeCostSetup

namespace solvers.«23».runtimeCost.Setup
open FormalSW.PyAst HeapCalls HeapCalls.Runtime
abbrev program := Solvers.P23Ast.module
abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)

def statements : List Stmt :=
  [.assign [.name "can"] (.builtinCall .bytearray
      [.binOp (.name "limit") .add (.constant (.int 1))]),
   .assign [.name "abund"] (.name "abundant"),
   .assign [.name "lim"] (.name "limit"),
   .assign [.name "L"] (.builtinCall .len [.name "abund"])]

def finalEnv (env : HeapCalls.Env) (limit count address buffer : Nat) : HeapCalls.Env :=
  setLocal (setLocal (setLocal (setLocal env "can" (.ref buffer))
    "abund" (.ref address)) "lim" (number limit)) "L" (number count)

/-- The actual bytearray constructor allocates and initializes every byte. -/
theorem allocate (limit : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (bound : env "limit" = some (number limit)) :
    EvaluatesIn .word program env (.builtinCall .bytearray
      [.binOp (.name "limit") .add (.constant (.int 1))]) heap (limit + 12)
      (.ref heap.length) (heap ++ [.bytearray (List.replicate (limit + 1) 0)]) := by
  have argument : EvaluatesIn .word program env
      (.binOp (.name "limit") .add (.constant (.int 1))) heap 6 (number (limit + 1)) heap := by
    apply EvaluatesIn.binary (work := 1) (EvaluatesIn.lookup bound) EvaluatesIn.constant
    · simp [Mutable.binary, scalarBinary, number]
    · rfl
  intro anchors stack
  have first : Reaches .word program anchors
      ⟨.expr (.builtinCall .bytearray [.binOp (.name "limit") .add (.constant (.int 1))]) env,
        stack, heap⟩ 1
      ⟨.expr (.binOp (.name "limit") .add (.constant (.int 1))) env,
        .arguments (.builtin .bytearray) [] [] env :: stack, heap⟩ := advance rfl rfl rfl
  have ready : Reaches .word program anchors
      ⟨.value (number (limit + 1)), .arguments (.builtin .bytearray) [] [] env :: stack, heap⟩ 1
      ⟨.operation (.builtin .bytearray) [number (limit + 1)], stack, heap⟩ := advance rfl rfl rfl
  have allocated : Reaches .word program anchors
      ⟨.operation (.builtin .bytearray) [number (limit + 1)], stack, heap⟩ (limit + 4)
      ⟨.value (.ref heap.length), stack, heap ++ [.bytearray (List.replicate (limit + 1) 0)]⟩ := by
    apply RuntimeCost.Amortized.Reaches.next (amount := limit + 4)
    · simp [step, result, run, primitive, Mutable.bytearray, Mutable.allocateBuffer,
        number, show ¬ (limit : Int) + 1 < 0 by omega]
      rfl
    · simp [charge, operation, BufferWork.operation, scalarNumber, PrimitiveWork.integer]
    · simp [potential, heapPotential, objectPotential]
  exact (((first.trans (argument anchors _)).trans ready).trans allocated).mono (by omega)

theorem source_run (items : List Nat) (limit address : Nat)
    (base : Mutable.Heap) (found : base[address]? = some (.list (Abundant.encode items)))
    (env : HeapCalls.Env) (getLimit : env "limit" = some (number limit))
    (getAbundant : env "abundant" = some (.ref address)) :
    BlockContinuesIn .word program env statements base (limit + 27)
      (finalEnv env limit items.length address base.length)
      (base ++ [.bytearray (List.replicate (limit + 1) 0)]) := by
  let e1 := setLocal env "can" (.ref base.length)
  let e2 := setLocal e1 "abund" (.ref address)
  let e3 := setLocal e2 "lim" (number limit)
  let heap := base ++ [.bytearray (List.replicate (limit + 1) 0)]
  have first := BlockContinuesIn.assign (name := "can") (allocate limit base env getLimit)
  have alias_run : BlockContinuesIn .word program e1 [.assign [.name "abund"] (.name "abundant")]
      heap 3 e2 heap := BlockContinuesIn.assign
    (EvaluatesIn.lookup (by simpa [e1, setLocal] using getAbundant))
  have bound : BlockContinuesIn .word program e2 [.assign [.name "lim"] (.name "limit")]
      heap 3 e3 heap := BlockContinuesIn.assign
    (EvaluatesIn.lookup (by simpa [e2, e1, setLocal] using getLimit))
  have array : heap[address]? = some (.list (Abundant.encode items)) := by
    simpa only [heap, List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1] using found
  have length : EvaluatesIn .word program e3 (.builtinCall .len [.name "abund"]) heap 5
      (number items.length) heap := by
    apply EvaluatesIn.builtin_single (builtin := .len) (work := 1) (by intro h; cases h) (by intro h; cases h)
      (EvaluatesIn.lookup (value := .ref address) (name := "abund") (by simp [e3, e2, setLocal]))
    · simp [primitive, Mutable.length, array, Abundant.encode, number]
    · simp [operation, sequenceLength, array]
  have last := BlockContinuesIn.assign (name := "L") length
  simpa only [statements, finalEnv, e3, e2, e1] using
    (first.append (alias_run.append (bound.append last))).mono (large := limit + 27) (by omega)

end solvers.«23».runtimeCost.Setup
