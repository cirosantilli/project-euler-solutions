import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row11471_small : List Nat := []
theorem row11471_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11471 row11471_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11471_block_005 : List Nat := []
theorem row11471_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11471 row11471_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11471_block_006 : List Nat := []
def plan11471_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 17), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 11), (.secondFactor 19), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 13), (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), (.secondFactor 19), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 11), (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 19), .first, .first, .second, .first, .first]
set_option maxRecDepth 2048 in
theorem row11471_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11471 row11471_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11471_block_006
  · decide
  · rfl

def row11471_block_007 : List Nat := [12569, 12647, 13679, 14057, 14249, 14369]
def plan11471_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 31), .first, .first, (.firstFactor 13), (.firstFactor 19), .first, (.firstFactor 7), (.secondFactor 29), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .possible, .first, (.firstFactor 7), (.secondFactor 29), (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 31), (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 13), (.firstFactor 7), (.secondFactor 23), .first, .first, .first, (.secondFactor 17), .first, .first, .second, .first, .first, (.firstFactor 7), .second, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), (.firstFactor 23), (.firstFactor 7), .second, .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .second, .first, (.secondFactor 7), (.firstFactor 29), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 37), .first, (.firstFactor 11), (.firstFactor 7), .first, .second, .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 19), .first, .first, (.firstFactor 11), .first, .possible, .first, .first, (.secondFactor 7), .first, (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 29), (.firstFactor 11), .first, (.firstFactor 13), .first, .second, .first, .second, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 31), (.secondFactor 7), .first, .second, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 37), .first, .second, .first, .first, (.firstFactor 13), .first, .possible, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 17), .first, (.firstFactor 11), .possible, .first, .first, .first, (.firstFactor 19), (.firstFactor 17), .first, .second, .first, .first, .possible, .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row11471_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11471 row11471_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11471_block_007
  · decide
  · rfl

def row11471_block_008 : List Nat := [14537]
def plan11471_block_008 : List Budget.Choice := [.first, (.firstFactor 23), .first, .first, .first, .possible, (.secondFactor 19), (.firstFactor 7), .first, .first, .second, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11471_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11471 row11471_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11471_block_008
  · decide
  · rfl

def row11471_block_009 : List Nat := []
theorem row11471_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11471 row11471_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11471_block_010 : List Nat := []
theorem row11471_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11471 row11471_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11471_block_011 : List Nat := []
theorem row11471_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11471 row11471_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11471_block_012 : List Nat := []
theorem row11471_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11471 row11471_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11471_block_013 : List Nat := []
theorem row11471_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11471 row11471_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11471_large_left_left : List Nat := row11471_small ++ row11471_block_005
theorem row11471_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11471 row11471_large_left_left :=
  Budget.RowUpperBound.branch row11471_small_covers row11471_block_005_covers

def row11471_large_left_right_right : List Nat := row11471_block_007 ++ row11471_block_008
theorem row11471_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11471 row11471_large_left_right_right :=
  Budget.RowUpperBound.branch row11471_block_007_covers row11471_block_008_covers

def row11471_large_left_right : List Nat := row11471_block_006 ++ row11471_large_left_right_right
theorem row11471_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11471 row11471_large_left_right :=
  Budget.RowUpperBound.branch row11471_block_006_covers row11471_large_left_right_right_covers

def row11471_large_left : List Nat := row11471_large_left_left ++ row11471_large_left_right
theorem row11471_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11471 row11471_large_left :=
  Budget.RowUpperBound.branch row11471_large_left_left_covers row11471_large_left_right_covers

def row11471_large_right_left : List Nat := row11471_block_009 ++ row11471_block_010
theorem row11471_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11471 row11471_large_right_left :=
  Budget.RowUpperBound.branch row11471_block_009_covers row11471_block_010_covers

def row11471_large_right_right_right : List Nat := row11471_block_012 ++ row11471_block_013
theorem row11471_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11471 row11471_large_right_right_right :=
  Budget.RowUpperBound.branch row11471_block_012_covers row11471_block_013_covers

def row11471_large_right_right : List Nat := row11471_block_011 ++ row11471_large_right_right_right
theorem row11471_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11471 row11471_large_right_right :=
  Budget.RowUpperBound.branch row11471_block_011_covers row11471_large_right_right_right_covers

def row11471_large_right : List Nat := row11471_large_right_left ++ row11471_large_right_right
theorem row11471_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11471 row11471_large_right :=
  Budget.RowUpperBound.branch row11471_large_right_left_covers row11471_large_right_right_covers

def row11471_large : List Nat := row11471_large_left ++ row11471_large_right
theorem row11471_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11471 row11471_large :=
  Budget.RowUpperBound.branch row11471_large_left_covers row11471_large_right_covers

def row11471 : List Nat := row11471_large
theorem row11471_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11471 row11471 := row11471_large_covers

def row11483_small : List Nat := []
theorem row11483_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11483 row11483_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11483_block_005 : List Nat := []
theorem row11483_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11483 row11483_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11483_block_006 : List Nat := [12041]
def plan11483_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 13), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, (.firstFactor 17), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, .second, (.secondFactor 23), (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 37), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .second, (.firstFactor 19), .first, (.firstFactor 7), .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 11), (.secondFactor 31), .first, .second, .first, .first, (.secondFactor 23), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 17)]
set_option maxRecDepth 2048 in
theorem row11483_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11483 row11483_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11483_block_006
  · decide
  · rfl

def row11483_block_007 : List Nat := [12611, 13421, 13499, 13619]
def plan11483_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .second, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 11), (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .second, .first, (.firstFactor 17), (.secondFactor 7), .first, .second, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .second, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 13), .first, .second, .first, (.firstFactor 7), .first, .first, .first, .possible, .first, (.secondFactor 31), (.secondFactor 17), (.firstFactor 11), (.firstFactor 19), .first, (.secondFactor 7), .possible, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .possible, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 23), .first, (.secondFactor 23), .first, .first, (.firstFactor 13), (.secondFactor 29), .first, (.secondFactor 17), (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 17), (.firstFactor 31), .first, (.secondFactor 31), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row11483_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11483 row11483_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11483_block_007
  · decide
  · rfl

def row11483_block_008 : List Nat := []
def plan11483_block_008 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 23), .first, (.secondFactor 23), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11483_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11483 row11483_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11483_block_008
  · decide
  · rfl

def row11483_block_009 : List Nat := []
theorem row11483_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11483 row11483_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11483_block_010 : List Nat := []
theorem row11483_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11483 row11483_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11483_block_011 : List Nat := []
theorem row11483_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11483 row11483_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11483_block_012 : List Nat := []
theorem row11483_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11483 row11483_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11483_block_013 : List Nat := []
theorem row11483_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11483 row11483_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11483_large_left_left : List Nat := row11483_small ++ row11483_block_005
theorem row11483_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11483 row11483_large_left_left :=
  Budget.RowUpperBound.branch row11483_small_covers row11483_block_005_covers

def row11483_large_left_right_right : List Nat := row11483_block_007 ++ row11483_block_008
theorem row11483_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11483 row11483_large_left_right_right :=
  Budget.RowUpperBound.branch row11483_block_007_covers row11483_block_008_covers

def row11483_large_left_right : List Nat := row11483_block_006 ++ row11483_large_left_right_right
theorem row11483_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11483 row11483_large_left_right :=
  Budget.RowUpperBound.branch row11483_block_006_covers row11483_large_left_right_right_covers

def row11483_large_left : List Nat := row11483_large_left_left ++ row11483_large_left_right
theorem row11483_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11483 row11483_large_left :=
  Budget.RowUpperBound.branch row11483_large_left_left_covers row11483_large_left_right_covers

def row11483_large_right_left : List Nat := row11483_block_009 ++ row11483_block_010
theorem row11483_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11483 row11483_large_right_left :=
  Budget.RowUpperBound.branch row11483_block_009_covers row11483_block_010_covers

def row11483_large_right_right_right : List Nat := row11483_block_012 ++ row11483_block_013
theorem row11483_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11483 row11483_large_right_right_right :=
  Budget.RowUpperBound.branch row11483_block_012_covers row11483_block_013_covers

def row11483_large_right_right : List Nat := row11483_block_011 ++ row11483_large_right_right_right
theorem row11483_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11483 row11483_large_right_right :=
  Budget.RowUpperBound.branch row11483_block_011_covers row11483_large_right_right_right_covers

def row11483_large_right : List Nat := row11483_large_right_left ++ row11483_large_right_right
theorem row11483_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11483 row11483_large_right :=
  Budget.RowUpperBound.branch row11483_large_right_left_covers row11483_large_right_right_covers

def row11483_large : List Nat := row11483_large_left ++ row11483_large_right
theorem row11483_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11483 row11483_large :=
  Budget.RowUpperBound.branch row11483_large_left_covers row11483_large_right_covers

def row11483 : List Nat := row11483_large
theorem row11483_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11483 row11483 := row11483_large_covers

def row11489_small : List Nat := []
theorem row11489_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11489 row11489_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11489_block_005 : List Nat := []
theorem row11489_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11489 row11489_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11489_block_006 : List Nat := [11597, 11783]
def plan11489_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, (.secondFactor 7), .first, .first, .possible, .first, (.firstFactor 11), .first, .second, .first, .second, .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 13), .first, .possible, (.secondFactor 7), (.firstFactor 23), (.secondFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .second, (.secondFactor 17), .first, .second, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 31), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 23), (.firstFactor 7), .first, (.secondFactor 17), (.secondFactor 31), (.secondFactor 19), (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 29), (.secondFactor 23), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row11489_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11489 row11489_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11489_block_006
  · decide
  · rfl

def row11489_block_007 : List Nat := [12899, 13127, 13229, 13709]
def plan11489_block_007 : List Budget.Choice := [.first, .first, .second, .first, .first, (.secondFactor 29), (.firstFactor 11), .first, (.firstFactor 23), (.secondFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 37), (.secondFactor 29), (.firstFactor 7), .first, (.secondFactor 7), .second, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .possible, .first, .first, (.firstFactor 31), .first, (.firstFactor 29), (.firstFactor 11), .second, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 23), (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 19), (.secondFactor 13), .possible, .first, .second, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .possible, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 7), (.firstFactor 23), (.secondFactor 37), (.firstFactor 11), .first, .first, (.firstFactor 17), .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 19), .first, .first, (.secondFactor 29), .first, .second, .possible, .first, (.secondFactor 7), .first, .first, (.firstFactor 37), (.secondFactor 23), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .second, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 13), .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 29), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 31), (.secondFactor 17), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 13), (.secondFactor 29), .first, .first, .first, .second, .first, (.firstFactor 23), .first, .first, (.secondFactor 13), .first, .first]
set_option maxRecDepth 2048 in
theorem row11489_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11489 row11489_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11489_block_007
  · decide
  · rfl

def row11489_block_008 : List Nat := []
def plan11489_block_008 : List Budget.Choice := [.first, (.firstFactor 29), .first, (.firstFactor 17), .first, (.secondFactor 19), (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11489_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11489 row11489_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11489_block_008
  · decide
  · rfl

def row11489_block_009 : List Nat := []
theorem row11489_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11489 row11489_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11489_block_010 : List Nat := []
theorem row11489_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11489 row11489_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11489_block_011 : List Nat := []
theorem row11489_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11489 row11489_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11489_block_012 : List Nat := []
theorem row11489_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11489 row11489_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11489_block_013 : List Nat := []
theorem row11489_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11489 row11489_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11489_large_left_left : List Nat := row11489_small ++ row11489_block_005
theorem row11489_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11489 row11489_large_left_left :=
  Budget.RowUpperBound.branch row11489_small_covers row11489_block_005_covers

def row11489_large_left_right_right : List Nat := row11489_block_007 ++ row11489_block_008
theorem row11489_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11489 row11489_large_left_right_right :=
  Budget.RowUpperBound.branch row11489_block_007_covers row11489_block_008_covers

def row11489_large_left_right : List Nat := row11489_block_006 ++ row11489_large_left_right_right
theorem row11489_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11489 row11489_large_left_right :=
  Budget.RowUpperBound.branch row11489_block_006_covers row11489_large_left_right_right_covers

def row11489_large_left : List Nat := row11489_large_left_left ++ row11489_large_left_right
theorem row11489_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11489 row11489_large_left :=
  Budget.RowUpperBound.branch row11489_large_left_left_covers row11489_large_left_right_covers

def row11489_large_right_left : List Nat := row11489_block_009 ++ row11489_block_010
theorem row11489_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11489 row11489_large_right_left :=
  Budget.RowUpperBound.branch row11489_block_009_covers row11489_block_010_covers

def row11489_large_right_right_right : List Nat := row11489_block_012 ++ row11489_block_013
theorem row11489_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11489 row11489_large_right_right_right :=
  Budget.RowUpperBound.branch row11489_block_012_covers row11489_block_013_covers

def row11489_large_right_right : List Nat := row11489_block_011 ++ row11489_large_right_right_right
theorem row11489_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11489 row11489_large_right_right :=
  Budget.RowUpperBound.branch row11489_block_011_covers row11489_large_right_right_right_covers

def row11489_large_right : List Nat := row11489_large_right_left ++ row11489_large_right_right
theorem row11489_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11489 row11489_large_right :=
  Budget.RowUpperBound.branch row11489_large_right_left_covers row11489_large_right_right_covers

def row11489_large : List Nat := row11489_large_left ++ row11489_large_right
theorem row11489_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11489 row11489_large :=
  Budget.RowUpperBound.branch row11489_large_left_covers row11489_large_right_covers

def row11489 : List Nat := row11489_large
theorem row11489_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11489 row11489 := row11489_large_covers

def row11491_small : List Nat := []
theorem row11491_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11491 row11491_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11491_block_005 : List Nat := []
theorem row11491_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11491 row11491_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11491_block_006 : List Nat := [12409]
def plan11491_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 29), .first, .first, (.secondFactor 7), .first, (.secondFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 31), .first, .first, .first, .first, .first, .first, .second, .first, .first, (.firstFactor 13), .first, .first, .second, .first, .second, .first, .first, (.secondFactor 17), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 19), .first, (.secondFactor 13), .first, (.secondFactor 17), (.firstFactor 7), .first, .possible, .first]
set_option maxRecDepth 2048 in
theorem row11491_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11491 row11491_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11491_block_006
  · decide
  · rfl

def row11491_block_007 : List Nat := [13003, 13291]
def plan11491_block_007 : List Budget.Choice := [(.secondFactor 19), (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .second, (.secondFactor 17), (.firstFactor 37), (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 31), (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 37), .first, (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .possible, .first, (.firstFactor 11), (.firstFactor 37), .first, .first, .first, (.firstFactor 7), (.secondFactor 17), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 23), (.firstFactor 31), (.secondFactor 7), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .possible, (.secondFactor 17), (.secondFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 37), (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.secondFactor 23), (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 19), .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, (.firstFactor 11), (.firstFactor 17), .first, .first, (.firstFactor 11), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 31), (.firstFactor 37), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.secondFactor 31), .second, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row11491_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11491 row11491_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11491_block_007
  · decide
  · rfl

def row11491_block_008 : List Nat := []
def plan11491_block_008 : List Budget.Choice := [(.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11491_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11491 row11491_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11491_block_008
  · decide
  · rfl

def row11491_block_009 : List Nat := []
theorem row11491_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11491 row11491_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11491_block_010 : List Nat := []
theorem row11491_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11491 row11491_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11491_block_011 : List Nat := []
theorem row11491_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11491 row11491_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11491_block_012 : List Nat := []
theorem row11491_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11491 row11491_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11491_block_013 : List Nat := []
theorem row11491_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11491 row11491_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11491_large_left_left : List Nat := row11491_small ++ row11491_block_005
theorem row11491_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11491 row11491_large_left_left :=
  Budget.RowUpperBound.branch row11491_small_covers row11491_block_005_covers

def row11491_large_left_right_right : List Nat := row11491_block_007 ++ row11491_block_008
theorem row11491_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11491 row11491_large_left_right_right :=
  Budget.RowUpperBound.branch row11491_block_007_covers row11491_block_008_covers

def row11491_large_left_right : List Nat := row11491_block_006 ++ row11491_large_left_right_right
theorem row11491_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11491 row11491_large_left_right :=
  Budget.RowUpperBound.branch row11491_block_006_covers row11491_large_left_right_right_covers

def row11491_large_left : List Nat := row11491_large_left_left ++ row11491_large_left_right
theorem row11491_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11491 row11491_large_left :=
  Budget.RowUpperBound.branch row11491_large_left_left_covers row11491_large_left_right_covers

def row11491_large_right_left : List Nat := row11491_block_009 ++ row11491_block_010
theorem row11491_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11491 row11491_large_right_left :=
  Budget.RowUpperBound.branch row11491_block_009_covers row11491_block_010_covers

def row11491_large_right_right_right : List Nat := row11491_block_012 ++ row11491_block_013
theorem row11491_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11491 row11491_large_right_right_right :=
  Budget.RowUpperBound.branch row11491_block_012_covers row11491_block_013_covers

def row11491_large_right_right : List Nat := row11491_block_011 ++ row11491_large_right_right_right
theorem row11491_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11491 row11491_large_right_right :=
  Budget.RowUpperBound.branch row11491_block_011_covers row11491_large_right_right_right_covers

def row11491_large_right : List Nat := row11491_large_right_left ++ row11491_large_right_right
theorem row11491_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11491 row11491_large_right :=
  Budget.RowUpperBound.branch row11491_large_right_left_covers row11491_large_right_right_covers

def row11491_large : List Nat := row11491_large_left ++ row11491_large_right
theorem row11491_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11491 row11491_large :=
  Budget.RowUpperBound.branch row11491_large_left_covers row11491_large_right_covers

def row11491 : List Nat := row11491_large
theorem row11491_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11491 row11491 := row11491_large_covers

end ProjectEulerSolutions.PrimePairLarge
