import { createSlice } from '@reduxjs/toolkit'

const initialState = {
  video: null,
  banner: null,
}

const videoSlice = createSlice({
  name: 'video',
  initialState,
  reducers: {
    setVideo: (state, action) => {
      state.video = action.payload
    },
    setBanner: (state, action) => {
      state.banner = action.payload
    },
    clearBanner: (state) => {
      state.banner = null
    },
  },
})

export const { setVideo, setBanner, clearBanner } = videoSlice.actions

export default videoSlice.reducer

export const setBannerThunk = (message, type) => {
  return async (dispatch) => {
    dispatch(setBanner({ message, type }))
    setTimeout(() => {
      dispatch(clearBanner())
    }, 2000)
  }
}