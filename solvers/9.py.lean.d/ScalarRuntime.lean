import solvers.«9.py.lean.d».SortingRuntime
import lang.py.FormalSW.HeapRuntimeCostBooleans

namespace solvers.«9».runtimeCost
open FormalSW.PyAst HeapCalls.Runtime solvers.«9».runtimeSource
private abbrev program := Solvers.P9Ast.module

theorem integer_binary
    (left : EvaluatesIn .word program env a heap ca (.scalar (.int av)) middle)
    (right : EvaluatesIn .word program env b middle cb (.scalar (.int bv)) afterRight)
    (evaluated : Mutable.binary op (.scalar (.int av)) (.scalar (.int bv)) afterRight = some (.ok value, afterRight))
    (priced : PrimitiveWork.binary .word op (.int av) (.int bv) = some 1 := by rfl) :
    EvaluatesIn .word program env (.binOp a op b) heap (ca + cb + 4) value afterRight :=
  EvaluatesIn.binary left right evaluated priced

theorem integer_compare
    (left : EvaluatesIn .word program env a heap ca (.scalar (.int av)) middle)
    (right : EvaluatesIn .word program env b middle cb (.scalar (.int bv)) afterRight)
    (evaluated : HeapCalls.compare op (.scalar (.int av)) (.scalar (.int bv)) afterRight = some (.ok value, afterRight)) :
    EvaluatesIn .word program env (.compare a [(op, b)]) heap (ca + cb + 5) value afterRight :=
  EvaluatesIn.compare left right evaluated rfl

end solvers.«9».runtimeCost
