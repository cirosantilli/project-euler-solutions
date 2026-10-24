import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row10331_small : List Nat := []
theorem row10331_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10331 row10331_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10331_block_005 : List Nat := []
def plan10331_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 7), (.firstFactor 23), .first, .first]
set_option maxRecDepth 2048 in
theorem row10331_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10331 row10331_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan10331_block_005
  · decide
  · rfl

def row10331_block_006 : List Nat := [10499, 11411, 11489, 12347]
def plan10331_block_006 : List Budget.Choice := [(.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .possible, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 37), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 37), (.firstFactor 13), .first, (.firstFactor 11), .first, .second, .first, .first, (.secondFactor 23), (.firstFactor 7), .first, (.secondFactor 29), .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 31), .first, .second, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 37), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 19), .first, (.firstFactor 17), .first, .first, (.firstFactor 29), .possible, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), (.secondFactor 7), .possible, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 31), .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .second, (.secondFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .second, .first, (.firstFactor 11), (.secondFactor 7), .first, .second, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 31), .first, (.firstFactor 11), (.firstFactor 13), (.firstFactor 7), (.secondFactor 19), .first, (.secondFactor 29), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 19), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 23), .first, .possible, .first, (.firstFactor 11), .first, .first, .second, .first, (.firstFactor 23)]
set_option maxRecDepth 2048 in
theorem row10331_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10331 row10331_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10331_block_006
  · decide
  · rfl

def row10331_block_007 : List Nat := [12893, 12959, 13841, 14081]
def plan10331_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 17), .second, .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, .second, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .possible, (.firstFactor 7), .first, (.secondFactor 7), .second, .first, (.secondFactor 13), (.firstFactor 7), (.secondFactor 7), .possible, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .second, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, .second, .first, .second, (.firstFactor 23), .first, (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .second, .first, .first, (.secondFactor 7), .second, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.secondFactor 17), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 37), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .possible, (.secondFactor 13), .first, (.secondFactor 7), .first, .second, (.secondFactor 29), .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 23), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .possible, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 31), (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 37), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row10331_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10331 row10331_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10331_block_007
  · decide
  · rfl

def row10331_block_008 : List Nat := [14891]
def plan10331_block_008 : List Budget.Choice := [.first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 19), (.secondFactor 7), .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 13), .first, (.secondFactor 19), (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 19), .first, (.firstFactor 13), .first, .second, .first, (.secondFactor 23), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .second, .first, (.firstFactor 29), .first, .possible, (.firstFactor 23), .first, .first, (.firstFactor 17), .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 29), .first, .first, (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 13), (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 7), (.firstFactor 31), .first, .first, .second, .first, .second, .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 29), (.firstFactor 11), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), (.secondFactor 29), (.firstFactor 13), .first, (.secondFactor 13), .first, .second, .first, (.firstFactor 13), .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10331_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10331 row10331_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10331_block_008
  · decide
  · rfl

def row10331_block_009 : List Nat := []
theorem row10331_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10331 row10331_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10331_block_010 : List Nat := []
theorem row10331_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10331 row10331_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10331_block_011 : List Nat := []
theorem row10331_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10331 row10331_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10331_block_012 : List Nat := []
theorem row10331_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10331 row10331_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10331_block_013 : List Nat := []
theorem row10331_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10331 row10331_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10331_large_left_left : List Nat := row10331_small ++ row10331_block_005
theorem row10331_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10331 row10331_large_left_left :=
  Budget.RowUpperBound.branch row10331_small_covers row10331_block_005_covers

def row10331_large_left_right_right : List Nat := row10331_block_007 ++ row10331_block_008
theorem row10331_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10331 row10331_large_left_right_right :=
  Budget.RowUpperBound.branch row10331_block_007_covers row10331_block_008_covers

def row10331_large_left_right : List Nat := row10331_block_006 ++ row10331_large_left_right_right
theorem row10331_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10331 row10331_large_left_right :=
  Budget.RowUpperBound.branch row10331_block_006_covers row10331_large_left_right_right_covers

def row10331_large_left : List Nat := row10331_large_left_left ++ row10331_large_left_right
theorem row10331_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10331 row10331_large_left :=
  Budget.RowUpperBound.branch row10331_large_left_left_covers row10331_large_left_right_covers

def row10331_large_right_left : List Nat := row10331_block_009 ++ row10331_block_010
theorem row10331_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10331 row10331_large_right_left :=
  Budget.RowUpperBound.branch row10331_block_009_covers row10331_block_010_covers

def row10331_large_right_right_right : List Nat := row10331_block_012 ++ row10331_block_013
theorem row10331_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10331 row10331_large_right_right_right :=
  Budget.RowUpperBound.branch row10331_block_012_covers row10331_block_013_covers

def row10331_large_right_right : List Nat := row10331_block_011 ++ row10331_large_right_right_right
theorem row10331_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10331 row10331_large_right_right :=
  Budget.RowUpperBound.branch row10331_block_011_covers row10331_large_right_right_right_covers

def row10331_large_right : List Nat := row10331_large_right_left ++ row10331_large_right_right
theorem row10331_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10331 row10331_large_right :=
  Budget.RowUpperBound.branch row10331_large_right_left_covers row10331_large_right_right_covers

def row10331_large : List Nat := row10331_large_left ++ row10331_large_right
theorem row10331_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10331 row10331_large :=
  Budget.RowUpperBound.branch row10331_large_left_covers row10331_large_right_covers

def row10331 : List Nat := row10331_large
theorem row10331_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10331 row10331 := row10331_large_covers

def row10333_small : List Nat := []
theorem row10333_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10333 row10333_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10333_block_005 : List Nat := []
def plan10333_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.secondFactor 7), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row10333_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10333 row10333_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan10333_block_005
  · decide
  · rfl

def row10333_block_006 : List Nat := [10939, 11047, 11311]
def plan10333_block_006 : List Budget.Choice := [.first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 29), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 23), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 17), .second, (.firstFactor 11), .first, .first, (.secondFactor 17), (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .second, .first, .first, .first, (.firstFactor 23), .first, .first, .possible, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .possible, .first, (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 13), .first, (.secondFactor 13), .first, (.firstFactor 37), .first, (.secondFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 19), .first, .possible, (.firstFactor 23), .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 31), (.secondFactor 29), .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 17), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .second, (.firstFactor 19), .first, (.firstFactor 17), (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.secondFactor 7), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 37), (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 13), .first, .first, .second, .first, .first, (.firstFactor 13), .first, (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first]
set_option maxRecDepth 2048 in
theorem row10333_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10333 row10333_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10333_block_006
  · decide
  · rfl

def row10333_block_007 : List Nat := [12979, 13291, 13513, 13789, 14407, 14461]
def plan10333_block_007 : List Budget.Choice := [(.firstFactor 23), (.secondFactor 23), .first, .second, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 37), .first, .first, (.secondFactor 7), (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), (.secondFactor 17), (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .second, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .second, .first, (.firstFactor 11), (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .possible, .first, .first, (.secondFactor 7), .first, (.secondFactor 29), (.secondFactor 13), .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .possible, (.firstFactor 29), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 23), (.firstFactor 13), .second, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .possible, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 37), (.firstFactor 11), .first, (.firstFactor 17), .first, (.secondFactor 19), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .possible, .first, (.secondFactor 31), .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .second, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 37), .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 19), (.firstFactor 17), (.firstFactor 11), .first, .first, (.secondFactor 17), .first, (.firstFactor 29), (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 7), .possible, .first, .second, .first, (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 19), .possible]
set_option maxRecDepth 2048 in
theorem row10333_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10333 row10333_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10333_block_007
  · decide
  · rfl

def row10333_block_008 : List Nat := [15091, 15541]
def plan10333_block_008 : List Budget.Choice := [(.secondFactor 19), .first, (.firstFactor 13), .first, .second, .first, .first, .first, (.secondFactor 31), (.firstFactor 11), .first, (.firstFactor 19), .first, (.secondFactor 13), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .second, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .possible, .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 7), (.secondFactor 23), (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .second, .first, .first, (.secondFactor 7), (.secondFactor 19), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 23), .first, (.secondFactor 23), (.firstFactor 11), .first, (.firstFactor 13), (.secondFactor 13), .first, .first, (.firstFactor 37), .first, .first, .second, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .possible, .first, (.firstFactor 29), .first, .first, .first, .second, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 31), .first, (.firstFactor 11), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10333_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10333 row10333_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10333_block_008
  · decide
  · rfl

def row10333_block_009 : List Nat := []
theorem row10333_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10333 row10333_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10333_block_010 : List Nat := []
theorem row10333_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10333 row10333_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10333_block_011 : List Nat := []
theorem row10333_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10333 row10333_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10333_block_012 : List Nat := []
theorem row10333_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10333 row10333_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10333_block_013 : List Nat := []
theorem row10333_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10333 row10333_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10333_large_left_left : List Nat := row10333_small ++ row10333_block_005
theorem row10333_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10333 row10333_large_left_left :=
  Budget.RowUpperBound.branch row10333_small_covers row10333_block_005_covers

def row10333_large_left_right_right : List Nat := row10333_block_007 ++ row10333_block_008
theorem row10333_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10333 row10333_large_left_right_right :=
  Budget.RowUpperBound.branch row10333_block_007_covers row10333_block_008_covers

def row10333_large_left_right : List Nat := row10333_block_006 ++ row10333_large_left_right_right
theorem row10333_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10333 row10333_large_left_right :=
  Budget.RowUpperBound.branch row10333_block_006_covers row10333_large_left_right_right_covers

def row10333_large_left : List Nat := row10333_large_left_left ++ row10333_large_left_right
theorem row10333_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10333 row10333_large_left :=
  Budget.RowUpperBound.branch row10333_large_left_left_covers row10333_large_left_right_covers

def row10333_large_right_left : List Nat := row10333_block_009 ++ row10333_block_010
theorem row10333_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10333 row10333_large_right_left :=
  Budget.RowUpperBound.branch row10333_block_009_covers row10333_block_010_covers

def row10333_large_right_right_right : List Nat := row10333_block_012 ++ row10333_block_013
theorem row10333_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10333 row10333_large_right_right_right :=
  Budget.RowUpperBound.branch row10333_block_012_covers row10333_block_013_covers

def row10333_large_right_right : List Nat := row10333_block_011 ++ row10333_large_right_right_right
theorem row10333_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10333 row10333_large_right_right :=
  Budget.RowUpperBound.branch row10333_block_011_covers row10333_large_right_right_right_covers

def row10333_large_right : List Nat := row10333_large_right_left ++ row10333_large_right_right
theorem row10333_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10333 row10333_large_right :=
  Budget.RowUpperBound.branch row10333_large_right_left_covers row10333_large_right_right_covers

def row10333_large : List Nat := row10333_large_left ++ row10333_large_right
theorem row10333_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10333 row10333_large :=
  Budget.RowUpperBound.branch row10333_large_left_covers row10333_large_right_covers

def row10333 : List Nat := row10333_large
theorem row10333_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10333 row10333 := row10333_large_covers

def row10337_small : List Nat := []
theorem row10337_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10337 row10337_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10337_block_005 : List Nat := []
def plan10337_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row10337_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10337 row10337_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan10337_block_005
  · decide
  · rfl

def row10337_block_006 : List Nat := [11519, 11579, 11681]
def plan10337_block_006 : List Budget.Choice := [(.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 7), .second, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .second, .first, .first, .first, (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 17), .first, (.firstFactor 23), .first, (.firstFactor 37), .first, .first, (.secondFactor 7), (.firstFactor 19), .first, .second, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 29), .first, (.secondFactor 31), .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 17), .first, .first, .first, .possible, .first, (.firstFactor 13), .first, .possible, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .possible, .first, (.firstFactor 29), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 11), (.secondFactor 7), .first, (.secondFactor 23), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), (.secondFactor 37), .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 31), (.firstFactor 17), .first, (.firstFactor 23), .first, (.firstFactor 7), (.secondFactor 17), .first, (.secondFactor 23), (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 29), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 37), .first, (.secondFactor 19)]
set_option maxRecDepth 2048 in
theorem row10337_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10337 row10337_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10337_block_006
  · decide
  · rfl

def row10337_block_007 : List Nat := [13049, 13697, 13757, 13763, 13931]
def plan10337_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 31), (.firstFactor 19), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 29), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .possible, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 7), (.firstFactor 23), .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 37), .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 13), .first, .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .possible, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .possible, .first, .possible, (.firstFactor 17), .first, (.secondFactor 13), .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .second, .first, .possible, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 11), .second, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.secondFactor 23), .first, (.firstFactor 37), .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, .second, (.secondFactor 7), .first, (.secondFactor 37), .first, .first, (.firstFactor 23), .first, .first, .first, .first, .second, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row10337_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10337 row10337_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10337_block_007
  · decide
  · rfl

def row10337_block_008 : List Nat := [15017]
def plan10337_block_008 : List Budget.Choice := [.first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 37), (.firstFactor 11), .first, (.firstFactor 7), .second, .first, (.firstFactor 29), (.firstFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 13), (.secondFactor 17), (.firstFactor 7), .first, (.secondFactor 37), .first, (.firstFactor 23), .first, (.secondFactor 13), .first, .first, (.firstFactor 29), (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 29), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 7), .first, .first, .possible, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 19), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 23), (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 31), (.secondFactor 17), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 17), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 31), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10337_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10337 row10337_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10337_block_008
  · decide
  · rfl

def row10337_block_009 : List Nat := []
theorem row10337_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10337 row10337_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10337_block_010 : List Nat := []
theorem row10337_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10337 row10337_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10337_block_011 : List Nat := []
theorem row10337_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10337 row10337_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10337_block_012 : List Nat := []
theorem row10337_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10337 row10337_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10337_block_013 : List Nat := []
theorem row10337_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10337 row10337_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10337_large_left_left : List Nat := row10337_small ++ row10337_block_005
theorem row10337_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10337 row10337_large_left_left :=
  Budget.RowUpperBound.branch row10337_small_covers row10337_block_005_covers

def row10337_large_left_right_right : List Nat := row10337_block_007 ++ row10337_block_008
theorem row10337_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10337 row10337_large_left_right_right :=
  Budget.RowUpperBound.branch row10337_block_007_covers row10337_block_008_covers

def row10337_large_left_right : List Nat := row10337_block_006 ++ row10337_large_left_right_right
theorem row10337_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10337 row10337_large_left_right :=
  Budget.RowUpperBound.branch row10337_block_006_covers row10337_large_left_right_right_covers

def row10337_large_left : List Nat := row10337_large_left_left ++ row10337_large_left_right
theorem row10337_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10337 row10337_large_left :=
  Budget.RowUpperBound.branch row10337_large_left_left_covers row10337_large_left_right_covers

def row10337_large_right_left : List Nat := row10337_block_009 ++ row10337_block_010
theorem row10337_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10337 row10337_large_right_left :=
  Budget.RowUpperBound.branch row10337_block_009_covers row10337_block_010_covers

def row10337_large_right_right_right : List Nat := row10337_block_012 ++ row10337_block_013
theorem row10337_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10337 row10337_large_right_right_right :=
  Budget.RowUpperBound.branch row10337_block_012_covers row10337_block_013_covers

def row10337_large_right_right : List Nat := row10337_block_011 ++ row10337_large_right_right_right
theorem row10337_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10337 row10337_large_right_right :=
  Budget.RowUpperBound.branch row10337_block_011_covers row10337_large_right_right_right_covers

def row10337_large_right : List Nat := row10337_large_right_left ++ row10337_large_right_right
theorem row10337_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10337 row10337_large_right :=
  Budget.RowUpperBound.branch row10337_large_right_left_covers row10337_large_right_right_covers

def row10337_large : List Nat := row10337_large_left ++ row10337_large_right
theorem row10337_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10337 row10337_large :=
  Budget.RowUpperBound.branch row10337_large_left_covers row10337_large_right_covers

def row10337 : List Nat := row10337_large
theorem row10337_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10337 row10337 := row10337_large_covers

def row10343_small : List Nat := []
theorem row10343_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10343 row10343_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10343_block_005 : List Nat := []
def plan10343_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first]
set_option maxRecDepth 2048 in
theorem row10343_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10343 row10343_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan10343_block_005
  · decide
  · rfl

def row10343_block_006 : List Nat := [10601, 11177, 11273, 11351, 11717]
def plan10343_block_006 : List Budget.Choice := [(.firstFactor 19), .first, .first, .first, (.firstFactor 13), .first, .second, .first, (.secondFactor 13), .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .possible, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 19), (.secondFactor 13), (.secondFactor 7), .first, .first, (.secondFactor 23), (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.secondFactor 13), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .possible, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .possible, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .possible, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 19), .first, .first, (.firstFactor 13), .first, (.secondFactor 17), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 23), (.secondFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .possible, .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 37), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 31), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 19), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 19), (.secondFactor 19), .first, (.firstFactor 17), (.secondFactor 7), .second, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 23), (.secondFactor 31), .first, .second, .first, .first, (.firstFactor 11), (.firstFactor 19), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, (.secondFactor 23), .first, (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row10343_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10343 row10343_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10343_block_006
  · decide
  · rfl

def row10343_block_007 : List Nat := [12659, 13829]
def plan10343_block_007 : List Budget.Choice := [.first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 31), (.secondFactor 13), (.firstFactor 11), .possible, (.firstFactor 7), (.firstFactor 37), .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, .first, (.secondFactor 7), .first, .second, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 37), (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 23), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 29), (.firstFactor 7), .first, .first, .first, (.secondFactor 17), (.firstFactor 31), .first, (.firstFactor 19), .first, (.firstFactor 23), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, (.secondFactor 23), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .second, .second, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 23), .first, (.firstFactor 37), .first, .possible, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 29), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 17), (.secondFactor 19), .first, (.firstFactor 13), .first, (.secondFactor 13), .first, (.firstFactor 37), .second, .first, .first, .first, (.firstFactor 11), .second, .first, (.secondFactor 17), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row10343_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10343 row10343_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10343_block_007
  · decide
  · rfl

def row10343_block_008 : List Nat := [14627, 15473]
def plan10343_block_008 : List Budget.Choice := [.first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .second, .first, .first, (.firstFactor 7), .first, .second, .first, .first, .possible, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .second, (.firstFactor 13), .first, .first, .first, (.secondFactor 23), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 19), .first, (.firstFactor 17), .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 13), (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 29), .second, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .second, .second, .possible, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .first, .second, .first, (.firstFactor 17), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10343_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10343 row10343_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10343_block_008
  · decide
  · rfl

def row10343_block_009 : List Nat := []
theorem row10343_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10343 row10343_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10343_block_010 : List Nat := []
theorem row10343_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10343 row10343_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10343_block_011 : List Nat := []
theorem row10343_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10343 row10343_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10343_block_012 : List Nat := []
theorem row10343_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10343 row10343_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10343_block_013 : List Nat := []
theorem row10343_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10343 row10343_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10343_large_left_left : List Nat := row10343_small ++ row10343_block_005
theorem row10343_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10343 row10343_large_left_left :=
  Budget.RowUpperBound.branch row10343_small_covers row10343_block_005_covers

def row10343_large_left_right_right : List Nat := row10343_block_007 ++ row10343_block_008
theorem row10343_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10343 row10343_large_left_right_right :=
  Budget.RowUpperBound.branch row10343_block_007_covers row10343_block_008_covers

def row10343_large_left_right : List Nat := row10343_block_006 ++ row10343_large_left_right_right
theorem row10343_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10343 row10343_large_left_right :=
  Budget.RowUpperBound.branch row10343_block_006_covers row10343_large_left_right_right_covers

def row10343_large_left : List Nat := row10343_large_left_left ++ row10343_large_left_right
theorem row10343_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10343 row10343_large_left :=
  Budget.RowUpperBound.branch row10343_large_left_left_covers row10343_large_left_right_covers

def row10343_large_right_left : List Nat := row10343_block_009 ++ row10343_block_010
theorem row10343_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10343 row10343_large_right_left :=
  Budget.RowUpperBound.branch row10343_block_009_covers row10343_block_010_covers

def row10343_large_right_right_right : List Nat := row10343_block_012 ++ row10343_block_013
theorem row10343_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10343 row10343_large_right_right_right :=
  Budget.RowUpperBound.branch row10343_block_012_covers row10343_block_013_covers

def row10343_large_right_right : List Nat := row10343_block_011 ++ row10343_large_right_right_right
theorem row10343_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10343 row10343_large_right_right :=
  Budget.RowUpperBound.branch row10343_block_011_covers row10343_large_right_right_right_covers

def row10343_large_right : List Nat := row10343_large_right_left ++ row10343_large_right_right
theorem row10343_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10343 row10343_large_right :=
  Budget.RowUpperBound.branch row10343_large_right_left_covers row10343_large_right_right_covers

def row10343_large : List Nat := row10343_large_left ++ row10343_large_right
theorem row10343_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10343 row10343_large :=
  Budget.RowUpperBound.branch row10343_large_left_covers row10343_large_right_covers

def row10343 : List Nat := row10343_large
theorem row10343_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10343 row10343 := row10343_large_covers

end ProjectEulerSolutions.PrimePairLarge
