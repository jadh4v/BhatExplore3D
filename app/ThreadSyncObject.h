#pragma once
#include <atomic>
#include <condition_variable>
#include <mutex>

class ThreadSyncObject
{
public:
    /// Synchronizer object needs to know how many threads are there
    /// before hand. Functions will fail if new threads are added later.
    ThreadSyncObject(unsigned int numOfThreads);

    /// Synchronization barrier function.
    void Sync();

private:
    const unsigned int m_NumberOfThreads;
    std::mutex m_Mutex;
    std::condition_variable m_ConditionVariable;
    std::atomic<int> m_WaitCounter = 0;
};