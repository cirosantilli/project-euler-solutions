import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row10799_small : List Nat := []
theorem row10799_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10799 row10799_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10799_block_005 : List Nat := []
theorem row10799_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10799 row10799_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10799_block_006 : List Nat := [11681, 12203, 12269]
def plan10799_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 19), (.firstFactor 7), (.secondFactor 23), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 23), (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 23), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 37), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .second, .first, .first, (.secondFactor 13), .first, (.firstFactor 17), (.secondFactor 7), (.firstFactor 11), .first, .possible, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 23), (.firstFactor 13), .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 17), (.firstFactor 37), (.firstFactor 23), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), (.secondFactor 19), .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 13), (.secondFactor 37), .first, .first, .first, (.firstFactor 7), .possible, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), .possible, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 37), .first, .first, .second, .first, .second]
set_option maxRecDepth 2048 in
theorem row10799_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10799 row10799_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10799_block_006
  · decide
  · rfl

def row10799_block_007 : List Nat := [13577, 13829, 13907, 14057]
def plan10799_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .first, .first, (.firstFactor 29), (.firstFactor 17), .first, .first, .first, .second, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 17), (.firstFactor 11), (.secondFactor 17), (.firstFactor 7), (.firstFactor 19), (.secondFactor 13), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 17), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.secondFactor 37), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 31), .first, (.firstFactor 13), .first, (.firstFactor 29), .first, .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 37), (.firstFactor 7), .first, (.firstFactor 13), .first, .second, .first, .possible, .first, .first, (.firstFactor 31), (.secondFactor 23), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 19), (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 17), (.secondFactor 13), .first, (.firstFactor 31), .first, .possible, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .possible, (.firstFactor 13), .first, (.firstFactor 37), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 11), .first, .possible, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.secondFactor 19), .first, .second, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.secondFactor 13), .first, .first, (.secondFactor 37), (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row10799_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10799 row10799_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10799_block_007
  · decide
  · rfl

def row10799_block_008 : List Nat := []
def plan10799_block_008 : List Budget.Choice := [.first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 31), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 7), (.secondFactor 29), .first, (.firstFactor 23), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 17), .first, (.secondFactor 37), .first, .first, .second, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 29), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 31), (.secondFactor 13), .first, .second, (.firstFactor 7), .first, .second, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10799_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10799 row10799_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10799_block_008
  · decide
  · rfl

def row10799_block_009 : List Nat := []
theorem row10799_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10799 row10799_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10799_block_010 : List Nat := []
theorem row10799_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10799 row10799_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10799_block_011 : List Nat := []
theorem row10799_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10799 row10799_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10799_block_012 : List Nat := []
theorem row10799_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10799 row10799_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10799_block_013 : List Nat := []
theorem row10799_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10799 row10799_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10799_large_left_left : List Nat := row10799_small ++ row10799_block_005
theorem row10799_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10799 row10799_large_left_left :=
  Budget.RowUpperBound.branch row10799_small_covers row10799_block_005_covers

def row10799_large_left_right_right : List Nat := row10799_block_007 ++ row10799_block_008
theorem row10799_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10799 row10799_large_left_right_right :=
  Budget.RowUpperBound.branch row10799_block_007_covers row10799_block_008_covers

def row10799_large_left_right : List Nat := row10799_block_006 ++ row10799_large_left_right_right
theorem row10799_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10799 row10799_large_left_right :=
  Budget.RowUpperBound.branch row10799_block_006_covers row10799_large_left_right_right_covers

def row10799_large_left : List Nat := row10799_large_left_left ++ row10799_large_left_right
theorem row10799_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10799 row10799_large_left :=
  Budget.RowUpperBound.branch row10799_large_left_left_covers row10799_large_left_right_covers

def row10799_large_right_left : List Nat := row10799_block_009 ++ row10799_block_010
theorem row10799_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10799 row10799_large_right_left :=
  Budget.RowUpperBound.branch row10799_block_009_covers row10799_block_010_covers

def row10799_large_right_right_right : List Nat := row10799_block_012 ++ row10799_block_013
theorem row10799_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10799 row10799_large_right_right_right :=
  Budget.RowUpperBound.branch row10799_block_012_covers row10799_block_013_covers

def row10799_large_right_right : List Nat := row10799_block_011 ++ row10799_large_right_right_right
theorem row10799_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10799 row10799_large_right_right :=
  Budget.RowUpperBound.branch row10799_block_011_covers row10799_large_right_right_right_covers

def row10799_large_right : List Nat := row10799_large_right_left ++ row10799_large_right_right
theorem row10799_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10799 row10799_large_right :=
  Budget.RowUpperBound.branch row10799_large_right_left_covers row10799_large_right_right_covers

def row10799_large : List Nat := row10799_large_left ++ row10799_large_right
theorem row10799_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10799 row10799_large :=
  Budget.RowUpperBound.branch row10799_large_left_covers row10799_large_right_covers

def row10799 : List Nat := row10799_large
theorem row10799_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10799 row10799 := row10799_large_covers

def row10831_small : List Nat := []
theorem row10831_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10831 row10831_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10831_block_005 : List Nat := []
theorem row10831_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10831 row10831_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10831_block_006 : List Nat := [11119, 12049, 12073]
def plan10831_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 23), .first, .first, (.firstFactor 31), (.firstFactor 7), .second, .first, (.secondFactor 13), .first, .second, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .possible, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 19), (.secondFactor 31), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 23), .second, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 13), .possible, .first, .possible, (.firstFactor 29), .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .second, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 31), (.secondFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row10831_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10831 row10831_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10831_block_006
  · decide
  · rfl

def row10831_block_007 : List Nat := [13711, 14083, 14341]
def plan10831_block_007 : List Budget.Choice := [(.secondFactor 13), (.firstFactor 13), .first, (.secondFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 13), (.firstFactor 7), (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 13), (.secondFactor 17), (.firstFactor 19), (.secondFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 23), .first, .first, .first, (.secondFactor 7), .second, (.firstFactor 37), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 31), (.firstFactor 19), .first, .first, .second, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 23), (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .possible, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 31), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .possible, .first, (.secondFactor 31), (.secondFactor 29), (.firstFactor 13), .first, .first, .second, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 19), .first, .possible, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 37), .first, (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 7), (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row10831_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10831 row10831_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10831_block_007
  · decide
  · rfl

def row10831_block_008 : List Nat := [14629, 15091]
def plan10831_block_008 : List Budget.Choice := [(.secondFactor 31), .first, (.secondFactor 19), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 11), .first, .first, .possible, .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 19), .second, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 31), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 31), (.secondFactor 29), .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .possible, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .second, .second, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10831_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10831 row10831_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10831_block_008
  · decide
  · rfl

def row10831_block_009 : List Nat := []
theorem row10831_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10831 row10831_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10831_block_010 : List Nat := []
theorem row10831_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10831 row10831_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10831_block_011 : List Nat := []
theorem row10831_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10831 row10831_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10831_block_012 : List Nat := []
theorem row10831_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10831 row10831_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10831_block_013 : List Nat := []
theorem row10831_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10831 row10831_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10831_large_left_left : List Nat := row10831_small ++ row10831_block_005
theorem row10831_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10831 row10831_large_left_left :=
  Budget.RowUpperBound.branch row10831_small_covers row10831_block_005_covers

def row10831_large_left_right_right : List Nat := row10831_block_007 ++ row10831_block_008
theorem row10831_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10831 row10831_large_left_right_right :=
  Budget.RowUpperBound.branch row10831_block_007_covers row10831_block_008_covers

def row10831_large_left_right : List Nat := row10831_block_006 ++ row10831_large_left_right_right
theorem row10831_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10831 row10831_large_left_right :=
  Budget.RowUpperBound.branch row10831_block_006_covers row10831_large_left_right_right_covers

def row10831_large_left : List Nat := row10831_large_left_left ++ row10831_large_left_right
theorem row10831_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10831 row10831_large_left :=
  Budget.RowUpperBound.branch row10831_large_left_left_covers row10831_large_left_right_covers

def row10831_large_right_left : List Nat := row10831_block_009 ++ row10831_block_010
theorem row10831_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10831 row10831_large_right_left :=
  Budget.RowUpperBound.branch row10831_block_009_covers row10831_block_010_covers

def row10831_large_right_right_right : List Nat := row10831_block_012 ++ row10831_block_013
theorem row10831_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10831 row10831_large_right_right_right :=
  Budget.RowUpperBound.branch row10831_block_012_covers row10831_block_013_covers

def row10831_large_right_right : List Nat := row10831_block_011 ++ row10831_large_right_right_right
theorem row10831_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10831 row10831_large_right_right :=
  Budget.RowUpperBound.branch row10831_block_011_covers row10831_large_right_right_right_covers

def row10831_large_right : List Nat := row10831_large_right_left ++ row10831_large_right_right
theorem row10831_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10831 row10831_large_right :=
  Budget.RowUpperBound.branch row10831_large_right_left_covers row10831_large_right_right_covers

def row10831_large : List Nat := row10831_large_left ++ row10831_large_right
theorem row10831_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10831 row10831_large :=
  Budget.RowUpperBound.branch row10831_large_left_covers row10831_large_right_covers

def row10831 : List Nat := row10831_large
theorem row10831_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10831 row10831 := row10831_large_covers

def row10837_small : List Nat := []
theorem row10837_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10837 row10837_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10837_block_005 : List Nat := []
theorem row10837_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10837 row10837_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10837_block_006 : List Nat := [10909, 11887]
def plan10837_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, (.secondFactor 7), .second, .first, .first, (.secondFactor 19), (.firstFactor 11), .possible, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, .second, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.secondFactor 19), .first, .second, .first, (.firstFactor 13), .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.secondFactor 31), (.firstFactor 11), .second, (.firstFactor 13), .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 29), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 23), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 17), .first, (.firstFactor 29), .first, .first, (.secondFactor 23), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 37), (.firstFactor 7), (.firstFactor 13), .first, .possible, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 23), (.firstFactor 11), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 31), (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 13), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 17), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .first]
set_option maxRecDepth 2048 in
theorem row10837_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10837 row10837_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10837_block_006
  · decide
  · rfl

def row10837_block_007 : List Nat := [12841, 13009, 13291, 13723]
def plan10837_block_007 : List Budget.Choice := [(.firstFactor 11), .second, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 31), (.firstFactor 31), .first, .second, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 13), .first, .first, (.firstFactor 19), (.secondFactor 13), .possible, .second, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 23), .first, .first, .first, .first, .second, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .possible, (.firstFactor 13), .first, .first, .first, (.secondFactor 13), (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 31), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .possible, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 11), .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 17), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 37), .first, .first, .first, .first, .possible, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 29), .first, (.firstFactor 11), .first, (.firstFactor 37), .first, .first, (.firstFactor 11), .first, .second, .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 13), .first, (.secondFactor 17), (.firstFactor 19), .first, .first, (.secondFactor 13), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 7), .first, .first, (.secondFactor 29), (.firstFactor 13), .second, .first, .first, .first, .first, (.secondFactor 17), (.secondFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 7), .second, .first, (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 13), .first, (.secondFactor 31), (.firstFactor 31)]
set_option maxRecDepth 2048 in
theorem row10837_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10837 row10837_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10837_block_007
  · decide
  · rfl

def row10837_block_008 : List Nat := []
def plan10837_block_008 : List Budget.Choice := [.first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .second, (.secondFactor 7), .first, .second, .first, (.firstFactor 13), .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10837_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10837 row10837_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10837_block_008
  · decide
  · rfl

def row10837_block_009 : List Nat := []
theorem row10837_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10837 row10837_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10837_block_010 : List Nat := []
theorem row10837_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10837 row10837_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10837_block_011 : List Nat := []
theorem row10837_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10837 row10837_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10837_block_012 : List Nat := []
theorem row10837_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10837 row10837_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10837_block_013 : List Nat := []
theorem row10837_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10837 row10837_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10837_large_left_left : List Nat := row10837_small ++ row10837_block_005
theorem row10837_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10837 row10837_large_left_left :=
  Budget.RowUpperBound.branch row10837_small_covers row10837_block_005_covers

def row10837_large_left_right_right : List Nat := row10837_block_007 ++ row10837_block_008
theorem row10837_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10837 row10837_large_left_right_right :=
  Budget.RowUpperBound.branch row10837_block_007_covers row10837_block_008_covers

def row10837_large_left_right : List Nat := row10837_block_006 ++ row10837_large_left_right_right
theorem row10837_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10837 row10837_large_left_right :=
  Budget.RowUpperBound.branch row10837_block_006_covers row10837_large_left_right_right_covers

def row10837_large_left : List Nat := row10837_large_left_left ++ row10837_large_left_right
theorem row10837_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10837 row10837_large_left :=
  Budget.RowUpperBound.branch row10837_large_left_left_covers row10837_large_left_right_covers

def row10837_large_right_left : List Nat := row10837_block_009 ++ row10837_block_010
theorem row10837_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10837 row10837_large_right_left :=
  Budget.RowUpperBound.branch row10837_block_009_covers row10837_block_010_covers

def row10837_large_right_right_right : List Nat := row10837_block_012 ++ row10837_block_013
theorem row10837_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10837 row10837_large_right_right_right :=
  Budget.RowUpperBound.branch row10837_block_012_covers row10837_block_013_covers

def row10837_large_right_right : List Nat := row10837_block_011 ++ row10837_large_right_right_right
theorem row10837_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10837 row10837_large_right_right :=
  Budget.RowUpperBound.branch row10837_block_011_covers row10837_large_right_right_right_covers

def row10837_large_right : List Nat := row10837_large_right_left ++ row10837_large_right_right
theorem row10837_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10837 row10837_large_right :=
  Budget.RowUpperBound.branch row10837_large_right_left_covers row10837_large_right_right_covers

def row10837_large : List Nat := row10837_large_left ++ row10837_large_right
theorem row10837_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10837 row10837_large :=
  Budget.RowUpperBound.branch row10837_large_left_covers row10837_large_right_covers

def row10837 : List Nat := row10837_large
theorem row10837_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10837 row10837 := row10837_large_covers

def row10847_small : List Nat := []
theorem row10847_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10847 row10847_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10847_block_005 : List Nat := []
theorem row10847_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10847 row10847_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10847_block_006 : List Nat := [11057, 11213, 11681, 12263]
def plan10847_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 37), (.firstFactor 17), .first, .possible, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .possible, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 23), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 31), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .possible, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .second, .second, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 31), .second, .first, .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 23), (.secondFactor 7), (.firstFactor 29), (.firstFactor 17), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 31), (.firstFactor 13), .first, (.firstFactor 7), .first, .possible, (.firstFactor 23), .first, (.secondFactor 19), .first, .first, .second, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 29), .first, (.firstFactor 19)]
set_option maxRecDepth 2048 in
theorem row10847_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10847 row10847_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10847_block_006
  · decide
  · rfl

def row10847_block_007 : List Nat := [13037, 13229, 13499, 13859, 14153, 14207]
def plan10847_block_007 : List Budget.Choice := [.first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 29), (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 19), (.secondFactor 7), .first, .first, .first, .second, .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .possible, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 37), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 7), .first, .possible, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), .possible, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.secondFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .second, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .possible, .first, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 31), .first, (.secondFactor 23), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .first, .possible, (.secondFactor 13), .first, (.secondFactor 7), .first, .possible, .first, (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 23), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.secondFactor 17), .first, .first]
set_option maxRecDepth 2048 in
theorem row10847_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10847 row10847_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10847_block_007
  · decide
  · rfl

def row10847_block_008 : List Nat := [14879, 15077]
def plan10847_block_008 : List Budget.Choice := [.first, (.secondFactor 29), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .second, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .second, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 23), .second, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .second, .first, .second, .first, .possible, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 13), .first, .first, (.firstFactor 31), .first, .first, .first, .first, .possible, .first, .first, (.firstFactor 29), (.firstFactor 7), .first, (.secondFactor 19), (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 17), (.secondFactor 13), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10847_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10847 row10847_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10847_block_008
  · decide
  · rfl

def row10847_block_009 : List Nat := []
theorem row10847_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10847 row10847_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10847_block_010 : List Nat := []
theorem row10847_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10847 row10847_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10847_block_011 : List Nat := []
theorem row10847_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10847 row10847_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10847_block_012 : List Nat := []
theorem row10847_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10847 row10847_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10847_block_013 : List Nat := []
theorem row10847_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10847 row10847_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10847_large_left_left : List Nat := row10847_small ++ row10847_block_005
theorem row10847_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10847 row10847_large_left_left :=
  Budget.RowUpperBound.branch row10847_small_covers row10847_block_005_covers

def row10847_large_left_right_right : List Nat := row10847_block_007 ++ row10847_block_008
theorem row10847_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10847 row10847_large_left_right_right :=
  Budget.RowUpperBound.branch row10847_block_007_covers row10847_block_008_covers

def row10847_large_left_right : List Nat := row10847_block_006 ++ row10847_large_left_right_right
theorem row10847_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10847 row10847_large_left_right :=
  Budget.RowUpperBound.branch row10847_block_006_covers row10847_large_left_right_right_covers

def row10847_large_left : List Nat := row10847_large_left_left ++ row10847_large_left_right
theorem row10847_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10847 row10847_large_left :=
  Budget.RowUpperBound.branch row10847_large_left_left_covers row10847_large_left_right_covers

def row10847_large_right_left : List Nat := row10847_block_009 ++ row10847_block_010
theorem row10847_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10847 row10847_large_right_left :=
  Budget.RowUpperBound.branch row10847_block_009_covers row10847_block_010_covers

def row10847_large_right_right_right : List Nat := row10847_block_012 ++ row10847_block_013
theorem row10847_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10847 row10847_large_right_right_right :=
  Budget.RowUpperBound.branch row10847_block_012_covers row10847_block_013_covers

def row10847_large_right_right : List Nat := row10847_block_011 ++ row10847_large_right_right_right
theorem row10847_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10847 row10847_large_right_right :=
  Budget.RowUpperBound.branch row10847_block_011_covers row10847_large_right_right_right_covers

def row10847_large_right : List Nat := row10847_large_right_left ++ row10847_large_right_right
theorem row10847_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10847 row10847_large_right :=
  Budget.RowUpperBound.branch row10847_large_right_left_covers row10847_large_right_right_covers

def row10847_large : List Nat := row10847_large_left ++ row10847_large_right
theorem row10847_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10847 row10847_large :=
  Budget.RowUpperBound.branch row10847_large_left_covers row10847_large_right_covers

def row10847 : List Nat := row10847_large
theorem row10847_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10847 row10847 := row10847_large_covers

end ProjectEulerSolutions.PrimePairLarge
