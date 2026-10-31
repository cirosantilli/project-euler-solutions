import solvers.«83.py.lean.d».NeighborSyntax

namespace solvers.«83»
open FormalSW.PyAst HeapCalls
open ProjectEulerSolutions.TriangleBufferCalls (number)
set_option Elab.async false

inductive NeighborDirection where
  | up | down | left | right

namespace NeighborDirection

def guard : NeighborDirection → Expr
  | .up => .compare (.name "r") [(.gt, .constant (.int 0))]
  | .down => .compare rowAfter [(.lt, .name "n")]
  | .left => .compare (.name "c") [(.gt, .constant (.int 0))]
  | .right => .compare columnAfter [(.lt, .name "m")]

def row : NeighborDirection → Expr
  | .up => rowBefore
  | .down => rowAfter
  | _ => .name "r"

def column : NeighborDirection → Expr
  | .left => columnBefore
  | .right => columnAfter
  | _ => .name "c"

def index : NeighborDirection → Expr
  | .up => .binOp (.binOp rowBefore .mult (.name "m")) .add (.name "c")
  | .down => .binOp (.binOp rowAfter .mult (.name "m")) .add (.name "c")
  | .left => .binOp (.binOp rowOffset .add (.name "c")) .sub (.constant (.int 1))
  | .right => .binOp (.binOp rowOffset .add (.name "c")) .add (.constant (.int 1))

def targetRow (direction : NeighborDirection) (r : Nat) : Nat := match direction with
  | .up => r - 1
  | .down => r + 1
  | _ => r

def targetColumn (direction : NeighborDirection) (c : Nat) : Nat := match direction with
  | .left => c - 1
  | .right => c + 1
  | _ => c

def enabled (direction : NeighborDirection) (r c height width : Nat) : Prop := match direction with
  | .up => 0 < r
  | .down => r + 1 < height
  | .left => 0 < c
  | .right => c + 1 < width

instance (direction : NeighborDirection) (r c height width : Nat) : Decidable (enabled direction r c height width) := by
  cases direction <;> unfold enabled <;> infer_instance

theorem ordinary_row (direction : NeighborDirection) : ∀ a b c, direction.row ≠ .slice a b c := by
  cases direction <;> intros a b c impossible <;> cases impossible

theorem ordinary_column (direction : NeighborDirection) : ∀ a b c, direction.column ≠ .slice a b c := by
  cases direction <;> intros a b c impossible <;> cases impossible

theorem bounds (direction : NeighborDirection) (r c height width : Nat)
    (rowInside : r < height) (columnInside : c < width) (active : direction.enabled r c height width) :
    direction.targetRow r < height ∧ direction.targetColumn c < width ∧
      direction.targetRow r * width + direction.targetColumn c < height * width := by
  have coordinates : direction.targetRow r < height ∧ direction.targetColumn c < width := by
    cases direction <;> simp only [targetRow, targetColumn, enabled] at * <;> omega
  refine ⟨coordinates.1, coordinates.2, ?_⟩
  have upper := Nat.mul_le_mul_right width (show direction.targetRow r + 1 ≤ height by omega)
  rw [Nat.add_mul, Nat.one_mul] at upper
  omega

end NeighborDirection

theorem directions_source : neighborsCode =
    ([.up, .down, .left, .right] : List NeighborDirection).map
      (fun direction => neighborBranch direction.guard direction.index direction.row direction.column) := rfl

private theorem increment (heap : Mutable.Heap) (env : HeapCalls.Env) (name : String) (n : Nat)
    (lookup : env name = some (number n)) :
    Evaluates program env (.binOp (.name name) .add (.constant (.int 1))) heap (number (n + 1 : Nat)) heap :=
  Evaluates.binary (Evaluates.lookup lookup) Evaluates.constant (by simp [Mutable.binary, scalarBinary, number])

private theorem decrement (heap : Mutable.Heap) (env : HeapCalls.Env) (name : String) (n : Nat)
    (positive : 0 < n) (lookup : env name = some (number n)) :
    Evaluates program env (.binOp (.name name) .sub (.constant (.int 1))) heap (number (n - 1 : Nat)) heap :=
  Evaluates.binary (Evaluates.lookup lookup) Evaluates.constant
    (by simp [Mutable.binary, scalarBinary, number, Int.ofNat_sub (by omega : 1 ≤ n)])

theorem neighbor_guard (direction : NeighborDirection) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (r c height width : Nat)
    (localRow : env "r" = some (number r)) (localColumn : env "c" = some (number c))
    (localHeight : env "n" = some (number height)) (localWidth : env "m" = some (number width)) :
    Evaluates program env direction.guard heap (.scalar (.bool (decide (direction.enabled r c height width)))) heap := by
  cases direction with
  | up =>
      apply Evaluates.compare (Evaluates.lookup localRow) Evaluates.constant
      simp [HeapCalls.compare, scalarCompare, evalCmpOp, NeighborDirection.enabled]
  | down =>
      apply Evaluates.compare (increment heap env "r" r localRow) (Evaluates.lookup localHeight)
      simp [HeapCalls.compare, scalarCompare, evalCmpOp, NeighborDirection.enabled]
      omega
  | left =>
      apply Evaluates.compare (Evaluates.lookup localColumn) Evaluates.constant
      simp [HeapCalls.compare, scalarCompare, evalCmpOp, NeighborDirection.enabled]
  | right =>
      apply Evaluates.compare (increment heap env "c" c localColumn) (Evaluates.lookup localWidth)
      simp [HeapCalls.compare, scalarCompare, evalCmpOp, NeighborDirection.enabled]
      omega

theorem neighbor_row (direction : NeighborDirection) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (r c height width : Nat) (localRow : env "r" = some (number r))
    (active : direction.enabled r c height width) :
    Evaluates program env direction.row heap (number (direction.targetRow r)) heap := by
  cases direction with
  | up => exact decrement heap env "r" r active localRow
  | down => exact increment heap env "r" r localRow
  | left => exact Evaluates.lookup localRow
  | right => exact Evaluates.lookup localRow

theorem neighbor_column (direction : NeighborDirection) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (r c height width : Nat) (localColumn : env "c" = some (number c))
    (active : direction.enabled r c height width) :
    Evaluates program env direction.column heap (number (direction.targetColumn c)) heap := by
  cases direction with
  | up => exact Evaluates.lookup localColumn
  | down => exact Evaluates.lookup localColumn
  | left => exact decrement heap env "c" c active localColumn
  | right => exact increment heap env "c" c localColumn

private theorem affine_index (heap : Mutable.Heap) (env : HeapCalls.Env) (rowExpr : Expr) (r c width : Nat)
    (row : Evaluates program env rowExpr heap (number r) heap)
    (localColumn : env "c" = some (number c)) (localWidth : env "m" = some (number width)) :
    Evaluates program env (.binOp (.binOp rowExpr .mult (.name "m")) .add (.name "c"))
      heap (number (r * width + c : Nat)) heap := by
  have product : Evaluates program env (.binOp rowExpr .mult (.name "m")) heap (number (r * width : Nat)) heap :=
    Evaluates.binary row (Evaluates.lookup localWidth) (by simp [Mutable.binary, scalarBinary, number])
  exact Evaluates.binary product (Evaluates.lookup localColumn) (by simp [Mutable.binary, scalarBinary, number])

theorem neighbor_index (direction : NeighborDirection) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (r c height width : Nat)
    (localRow : env "r" = some (number r)) (localColumn : env "c" = some (number c))
    (localWidth : env "m" = some (number width)) (active : direction.enabled r c height width) :
    Evaluates program env direction.index heap
      (number (direction.targetRow r * width + direction.targetColumn c : Nat)) heap := by
  cases direction with
  | up =>
      exact affine_index heap env rowBefore (r - 1) c width
        (decrement heap env "r" r active localRow) localColumn localWidth
  | down =>
      exact affine_index heap env rowAfter (r + 1) c width
        (increment heap env "r" r localRow) localColumn localWidth
  | left =>
      apply Evaluates.binary (affine_index heap env (.name "r") r c width (Evaluates.lookup localRow) localColumn localWidth)
        Evaluates.constant
      have positive : 1 ≤ c := by exact active
      simp [Mutable.binary, scalarBinary, number, NeighborDirection.targetRow, NeighborDirection.targetColumn,
        Int.ofNat_sub positive, Int.add_sub_assoc]
  | right =>
      apply Evaluates.binary (affine_index heap env (.name "r") r c width (Evaluates.lookup localRow) localColumn localWidth)
        Evaluates.constant
      simp [Mutable.binary, scalarBinary, number, NeighborDirection.targetRow, NeighborDirection.targetColumn,
        Int.add_assoc]

end solvers.«83»
