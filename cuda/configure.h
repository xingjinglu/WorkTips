#ifndef _CONFIGURE_H
#define _CONFIGURE_H


enum WmmaShape{wmma_8=8, wmma_16=16, wmma_32=32};

class Configure{

  public:
    Configure(): wmma_m_(16), wmma_n_(16), wmma_k_(16), blk_m_(128),
    blk_n_(128), blk_k_(12), warp_size_(32), blk_warps_(8), blk_row_warps_(2), 
    blk_col_warps_(4), warp_row_tiles_(4), warp_col_tiles_(2){}
    ~Configure(){}

  public:
    // Cube shape (the smallest shape of tile)
    int wmma_m_;
    int wmma_n_;
    int wmma_k_;

    // Deprecated.
    int blk_m_;
    int blk_n_;
    int blk_k_;
    //
    int sm_tiles_;

    // Tile number/shape of block.
    int blk_tiles_;
    int blk_tiles_x_;  // num of row. 
    int blk_tiles_y_;  // num of col.
    int blk_tiles_z_;

    // Warps number/shape of block.
    int warp_size_; // 32 as efault.
    int blk_warps_;
    int blk_warps_x_; // blk_col_warps_
    int blk_warps_y_; // blk_row_warps_
    int blk_warps_z_;

    int blk_row_warps_;  // warps-num on each row
    int blk_col_warps_;

    // Mapping between tiles and warps.
    int warp_tiles_x_;
    int warp_tiles_y_;
    int warp_tilex_z_;

    int warp_row_tiles_;
    int warp_col_tiles_;

    //
    int chunk_k_;

    // Persist.
    int chunk_col_;
};

#endif
