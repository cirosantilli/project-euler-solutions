import solvers.«98.py.lean.d».Pattern

namespace solvers.«98».squares_with_length
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.AnagramSquares
open pattern_signature (program number)
set_option Elab.async false

abbrev boundsCode : List Stmt :=
  [.assign [.name "lo"] (.binOp (.constant (.int 10)) .pow (.binOp (.name "L") .sub (.constant (.int 1)))),
   .assign [.name "hi"] (.binOp (.binOp (.constant (.int 10)) .pow (.name "L")) .sub (.constant (.int 1))),
   .assign [.name "n0"] (.binOp (.call (.name "isqrt") [.binOp (.name "lo") .sub (.constant (.int 1))]) .add (.constant (.int 1))),
   .assign [.name "n1"] (.call (.name "isqrt") [.name "hi"])]

theorem source_prefix :
    (executableBody ((findFunctionIn program.body "squares_with_length").getD ([], [])).2).take 4 = boundsCode := rfl

def ready (env : HeapCalls.Env) (width : Nat) : HeapCalls.Env :=
  setLocal (setLocal (setLocal (setLocal env "lo" (number (10 ^ (width - 1))))
    "hi" (number (10 ^ width - 1))) "n0" (number (firstRoot width))) "n1" (number (lastRoot width))

theorem isqrt_expression (n : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env) (expression : Expr)
    (evaluated : Evaluates program env expression heap (number n) heap) :
    Evaluates program env (.call (.name "isqrt") [expression]) heap (number n.sqrt) heap := by
  apply Evaluates.library_call (.cons evaluated .nil) rfl
  have root : Math.isqrt n = n.sqrt := Nat.eq_sqrt.mpr (Math.isqrt_spec n)
  have imported : libraryFunction program "isqrt" = some Math.isqrtCall := rfl
  simp [library, imported, scalarResult, Math.isqrtCall, root, number, show ¬(n : Int) < 0 by omega]

theorem prepare (width : Nat) (positive : 0 < width) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getWidth : env "L" = some (number width)) :
    BlockContinues program env boundsCode heap (ready env width) heap := by
  have lowerPositive : 0 < 10 ^ (width - 1) := Nat.pow_pos (by decide)
  have upperPositive : 0 < 10 ^ width := Nat.pow_pos (by decide)
  have difference : (width : Int) - 1 = (width - 1 : Nat) := by omega
  have lowerDifference : ((10 : Nat) ^ (width - 1) : Int) - 1 = (10 ^ (width - 1) - 1 : Nat) := by omega
  have upperDifference : ((10 : Nat) ^ width : Int) - 1 = (10 ^ width - 1 : Nat) := by omega
  let withLower := setLocal env "lo" (number (10 ^ (width - 1)))
  let withUpper := setLocal withLower "hi" (number (10 ^ width - 1))
  let withFirst := setLocal withUpper "n0" (number (firstRoot width))
  have lower : Evaluates program env
      (.binOp (.constant (.int 10)) .pow (.binOp (.name "L") .sub (.constant (.int 1))))
      heap (number (10 ^ (width - 1))) heap := by
    apply Evaluates.binary (av := number 10) (bv := number (width - 1)) (middle := heap) (afterRight := heap) Evaluates.constant
    · exact Evaluates.binary (av := number width) (bv := number 1) (Evaluates.lookup getWidth) Evaluates.constant
        (by simp [Mutable.binary, scalarBinary, number, difference])
    · simp [Mutable.binary, scalarBinary, number, show ¬((width - 1 : Nat) : Int) < 0 by omega]
  have upper : Evaluates program withLower
      (.binOp (.binOp (.constant (.int 10)) .pow (.name "L")) .sub (.constant (.int 1)))
      heap (number (10 ^ width - 1)) heap := by
    apply Evaluates.binary (av := number (10 ^ width)) (bv := number 1) (middle := heap) (afterRight := heap)
    · apply Evaluates.binary (av := number 10) (bv := number width) Evaluates.constant
        (Evaluates.lookup (by simpa [withLower, setLocal] using getWidth))
      simp [Mutable.binary, scalarBinary, number, show ¬(width : Int) < 0 by omega]
    · exact Evaluates.constant
    · simpa [Mutable.binary, scalarBinary, number] using upperDifference
  have first : Evaluates program withUpper
      (.binOp (.call (.name "isqrt") [.binOp (.name "lo") .sub (.constant (.int 1))]) .add (.constant (.int 1)))
      heap (number (firstRoot width)) heap := by
    apply Evaluates.binary (av := number (10 ^ (width - 1) - 1).sqrt) (bv := number 1)
      (middle := heap) (afterRight := heap)
    · apply isqrt_expression
      exact Evaluates.binary (av := number (10 ^ (width - 1))) (bv := number 1)
        (Evaluates.lookup (by simp [withUpper, withLower, setLocal])) Evaluates.constant
        (by simpa [Mutable.binary, scalarBinary, number] using lowerDifference)
    · exact Evaluates.constant
    · simp [Mutable.binary, scalarBinary, number, firstRoot]
  have last : Evaluates program withFirst (.call (.name "isqrt") [.name "hi"])
      heap (number (lastRoot width)) heap :=
    isqrt_expression (10 ^ width - 1) heap withFirst _ (Evaluates.lookup (by simp [withFirst, withUpper, setLocal]))
  exact (BlockContinues.assign lower).append ((BlockContinues.assign upper).append
    ((BlockContinues.assign first).append (BlockContinues.assign last)))

end solvers.«98».squares_with_length
